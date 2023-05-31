
tf.setBackend('webgl')
function dir_kernel(grid) {
 // Define the weights of the directional kernel
 return tf.tidy(() => {let boolgrid = tf.notEqual(grid,0).cast('int32')
 const kern_weights = tf.tensor([
        [[ 0, 0, 0],
        [ 0, -1, 0],
        [ 0, 0, 0]],
       [[    0,   -1,     0],
        [   -1, 6,   -1],
        [    0,   -1,     0]],

       [[ 0,     0, 0],
        [    0,    -1,     0],
        [ 0,     0, 0]]], [3, 3, 3], 'int32');
 const kernel = tf.reshape(kern_weights, [3, 3, 3, 1, 1])
 // Remove the extra dimensions and return the edge detected grid
 let dirgrid = tf.conv3d(tf.reshape(boolgrid,[1,grid.shape[0],grid.shape[1],grid.shape[2],1]), kernel, [1,1,1,1,1], 'same').reshape(grid.shape)
 return  dirgrid.clipByValue(0,1).cast('int32')});
}


function updatescalar(indices,scalar){
  for (let index = 0; index < indices.length; index++) {
    const local = indices[index]
    let meshindex = local[2]+grid.shape[0]*local[1]+local[0]*grid.shape[0]*grid.shape[1]
    mesh.getMatrixAt(meshindex, cubeProxy.matrix)
    cubeProxy.position.set(local[0]-offset[0], local[1]-offset[1], local[2]-offset[2]);
    cubeProxy.scale.setScalar(scalar)
    cubeProxy.updateMatrix()
    mesh.setMatrixAt(meshindex, cubeProxy.matrix)
  }
}

function updatevalues(indices,values){
for (let index = 0; index < indices.length; index++) {
  const local = indices[index]
  let meshindex = local[2]+grid.shape[0]*local[1]+local[0]*grid.shape[0]*grid.shape[1]
  mesh.setColorAt(meshindex, new THREE.Color(0, values[index], 0));
}}



function generateVoxelgrid() {
mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial(), grid.shape.reduce((a, v) => a * v))
offset = [grid.shape[0] / 2 - 0.5, grid.shape[1] / 2 - 0.5, grid.shape[2] / 2 - 0.5];
  for (let i = 0; i < grid.shape[0]; i++) {
    for (let j = 0; j < grid.shape[1]; j++) {
      for (let k = 0; k < grid.shape[2]; k++) {
    cubeProxy.position.set(i-offset[0], j-offset[1], k-offset[2]);
    cubeProxy.scale.setScalar(0)
    cubeProxy.updateMatrix()
    const index = k+grid.shape[0]*j+i*grid.shape[0]*grid.shape[1]
    mesh.setMatrixAt(index, cubeProxy.matrix)
  }}}
  mesh.setColorAt(0, new THREE.Color(0, 0, 0));
  scene.add(mesh);
}


async function  updateVoxelGrid() {
    let new_dir_grid = tf.tidy(() => {return dir_kernel(grid)})
    const diffgrid = tf.sub(new_dir_grid,dirGrid)
    // Cells to be deleted
    const delgrid = tf.equal(diffgrid,-1)
    let deldirIndices = await tf.whereAsync(delgrid)
    let delIndex = deldirIndices.arraySync()

    // Cells to be who face array needs to be Updated
    // Cells Generated
    const newgrid = tf.equal(diffgrid,1)
    let newdirIndices = await tf.whereAsync(newgrid)
    let newIndex  = newdirIndices.arraySync()
    let cellgrid = tf.notEqual(new_dir_grid,0)
    let cellindices = await tf.whereAsync(cellgrid)
    const cellvalues = tf.gatherND(grid, cellindices)
    let values = cellvalues.arraySync()
    cellIndex = cellindices.arraySync()
    // Update the cells that are generated
    updatescalar(newIndex,1)
    updatescalar(delIndex,0)
    updatevalues(cellIndex,values)
    mesh.instanceMatrix.needsUpdate = true;
    mesh.instanceColor.needsUpdate = true;
    dirGrid.assign(new_dir_grid);
    tf.dispose([
      deldirIndices,
      newdirIndices,
      cellindices,
      cellvalues,
      diffgrid,
      new_dir_grid,
      delgrid,
      newgrid,
      cellgrid
    ]);
  }

function transposecomplex(tensor,perms){
  return tf.tidy(()=>{
    const size = perms.map(index => tensor.shape[index])
    let real = tf.real(tensor).reshape(size).transpose(perms)
    let imag = tf.imag(tensor).reshape(size).transpose(perms)
    return tf.complex(real,imag)
  })
  }

function fft1(tensor,perms){
  return  tf.tidy(() => {
    let fttensor = tf.spectral.fft(tensor.transpose(perms))
return transposecomplex(fttensor,perms)})}

function fftn(tensor){
  return tf.tidy(() => {
let perms = tf.range(0,tensor.shape.length,1).arraySync()
let fttensor = tf.tidy(() => {return fft1(tensor,perms)})
for (let i=1;i<perms.length;i++){
const size = perms.length-1
const permsi = perms.slice();
[permsi[size],permsi[size-i]] = [permsi[size-i],permsi[size]];
fttensor = tf.tidy(() => {return fft1(fttensor,permsi)})}
return fttensor
})}

function ifft1(tensor,perms){
return  tf.tidy(() => {
let fttensor = tf.spectral.ifft(tensor.transpose(perms))
return transposecomplex(fttensor,perms)})}

function ifftn(tensor){
return tf.tidy(() => {
let perms = tf.range(0,tensor.shape.length,1).arraySync()
let fttensor = tf.tidy(() => {return ifft1(tensor,perms)})
for (let i=1;i<perms.length;i++){
const size = perms.length-1
const permsi = perms.slice();
[permsi[size],permsi[i-1]] = [permsi[i-1],permsi[size]];
fttensor = tf.tidy(() => {return ifft1(fttensor,permsi)})
}
return fttensor})}

function generate_kernel(radius, grid_size) {
  kernel =   tf.tidy(() => {
  const mid = Math.floor(grid_size/2);
  const bell = (x, m, s) => tf.exp(tf.neg(tf.pow(tf.div(tf.sub(x, m), s), 2)).div(2));
  const x = tf.range(-mid, mid,1);
  let [X, Y] = tf.meshgrid(x, x);
  X = X.expandDims(0).tile([X.shape[0],1, 1]);
  Y = Y.expandDims(0).tile([Y.shape[0],1, 1])
  const Z = X.transpose();
  const D = tf.div(tf.sqrt(tf.add(tf.add(tf.pow(X, 2), tf.pow(Y, 2)), tf.pow(Z, 2))), radius);
  const K = tf.mul(tf.cast(tf.less(D, 1), 'float32'), bell(D, 0.5, 0.15));
  return fftn(roll(K.div(tf.sum(K)), [0,1,2], [mid,mid,mid]).cast('complex64'))})
  }
  
function roll(grid, axis, shift) {
  return tf.tidy(() => {
  const grid_size = tf.tensor(grid.shape).gather(axis).dataSync();
  const limits = [grid_size[0]-shift[0], grid_size[1]-shift[1], grid_size[2]-shift[2]];
  const [a1, b1] = tf.split(grid, [limits[0], shift[0]], axis[0]);
  const x1 = tf.concat([b1,a1],axis[0]);
  const [a2, b2] = tf.split(x1, [limits[1], shift[1]], axis[1]);
  const x2 = tf.concat([b2,a2],axis[1])
  const [a3, b3] = tf.split(x2, [limits[2], shift[2]], axis[2]);
  return  tf.concat([b3,a3],axis[2])})
  }


  function golupdate() {  
    const dummy  = tf.tidy(() => {
    const U = tf.real(ifftn(tf.mul(kernel,fftn(grid.cast('complex64'))))).reshape(grid.shape)
     return tf.add(grid,tf.div(tf.tidy(()=> { return growth(U, m, s)}), T)).clipByValue(0, 1)})
    grid.assign(dummy)
    dummy.dispose()
    }
  function growth(U, m, s) {
    return tf.tidy(() => { return tf.sub(tf.mul(tf.exp(tf.sub(0,tf.div(U.sub(m).div(s).pow(2),2))),2),1)})
  }
  
  function generategrid(){
    const mid = Math.floor((grid_size -cluster_size)/ 2)
    grid = tf.tidy(() => {return tf.variable(tf.pad(tf.randomUniform([cluster_size,cluster_size,cluster_size]),[[mid,mid],[mid,mid],[mid,mid]]).cast('float32'))})
    dirGrid = tf.tidy(() => {return tf.variable(tf.zeros(grid.shape).cast('int32'))})}



  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
  const renderer = new THREE.WebGLRenderer();
  renderer.setSize( window.innerWidth, window.innerHeight );
  document.body.appendChild( renderer.domElement );


      let grid_size = 50
      let cluster_size = 20
      let m = 0.15
      let s = 0.017
      let T = 10
      let radius = 12
let grid = null, dirGrid = null, kernel = null, mesh = null, offset = null, cubeProxy = new THREE.Object3D()
generategrid()
generateVoxelgrid()
generate_kernel(radius, grid_size)
updateVoxelGrid() 
camera.position.x = 2*cluster_size ;
camera.position.y = 2*cluster_size ;
camera.position.z = 2*cluster_size ;
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; 
controls.dampingFactor = 0.05; // Set the damping factor for the damping effect
scene.background = new THREE.Color(0x101214);
const animate = function () {
  requestAnimationFrame( animate );
  controls.update(); // Update controls
  renderer.render( scene, camera );
};
animate();
let isRunning = false; // Variable to hold the interval ID

function animatestate() {
  if (isRunning) {
    // Update the grid every 1/T seconds
    setTimeout(function() {
        golupdate();
        updateVoxelGrid()
        animatestate();
      }, Math.floor(1000/T))

    // Render your game objects here
  }
}

// Function to start the game
function startstate() {
isRunning = true;
animatestate();
}

// Function to stop the game
function stopstate() {
isRunning = false;
}

function updatestate() {
  if (isRunning) {
    stopstate();
  } else {
    startstate();
  }
}
// Event listener for keydown event
document.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    updatestate();
  }
  
});;

document.addEventListener('DOMContentLoaded', function() {
  var menuItems = document.querySelectorAll('.menu li a');
  var containers = document.querySelectorAll('.container');

  menuItems.forEach(function(item, index) {
    item.addEventListener('click', function(e) {
      e.preventDefault();
      var container = containers[index];

      if (container.style.display === 'block') {
        container.style.display = 'none';
      } else {
        containers.forEach(function(c) {
          c.style.display = 'none';
        });
        container.style.display = 'block';
      }
    });
  });
});
// Get the DOM elements for display
const gridsizecounter = document.getElementById("gridsize");
const radiuscounter = document.getElementById("radius");
const timecounter = document.getElementById("time");
const gammacounter = document.getElementById("gamma");
const mucounter = document.getElementById("mu");
// Get the buttons for adding and subtracting
const gridsizeslider = document.getElementById("gridsize-slider");
const radiusAddButton = document.getElementById("radius-add-button");
const radiusSubButton = document.getElementById("radius-sub-button");
const timeAddButton = document.getElementById("time-add-button");
const timeSubButton = document.getElementById("time-sub-button");
const muAddButton = document.getElementById("mu-add-button");
const muSubButton = document.getElementById("mu-sub-button");
const sigmaAddButton = document.getElementById("sigma-add-button");
const sigmaSubButton = document.getElementById("sigma-sub-button");
const startstop = document.getElementById("startstop");
const reset = document.getElementById("reset");
let new_grid_size = grid_size
let gridrange = [2,100,49]
let rrange = [0,100,100]
let trange = [0,20,20]
let mrange = [0,1,20]
let srange = [0,1,50]
updategridsize();
updateradius();
updatet();
updatem();
updates();
gridsizeslider.value = grid_size
// Add event listeners to the buttons
gridsizeslider.addEventListener("change", () => {
  new_grid_size = gridsizeslider.value
  updategridsize();
});
radiusAddButton.addEventListener("click", () => {
  radius = inc(radius, rrange,1)
  updateradius();
});

radiusSubButton.addEventListener("click", () => {
  radius = inc(radius, rrange,-1)
  updateradius();
});

timeAddButton.addEventListener("click", () => {
  T = inc(T, trange,1)
  updatet();
});

timeSubButton.addEventListener("click", () => {
  T = inc(T, trange,-1)
  updatet();
});

muAddButton.addEventListener("click", () => {
  m = inc(m, mrange,1)
  updatem();
});

muSubButton.addEventListener("click", () => {
  m = inc(m, mrange,-1)
  updatem();
});

sigmaAddButton.addEventListener("click", () => {
  s = inc(m, srange,1)
  updates();
});

sigmaSubButton.addEventListener("click", () => {
  s = inc(s, srange,-1)
  updates();
});
reset.addEventListener("click", () => {
  grid.dispose()
  updateVoxelGrid()
  generategrid()
});
  function updategridsize() {
    let padding  = (new_grid_size-grid_size)/2
    prevstate = isRunning
    if (padding != 0) { 
      stopstate()
    if (padding > 0) {

      new_grid = tf.tidy(() => {return tf.variable(tf.pad(grid,[[padding,padding],[padding,padding],[padding,padding]]))})
      new_dir_grid = tf.tidy(() => {return tf.variable(tf.zeros(new_grid.shape).cast('int32'))})}
    else { 
      padding = -padding
      new_grid = tf.tidy(()=>{return tf.variable(grid.slice([padding,padding,padding],[grid_size-padding-1,grid_size-padding-1,grid_size-padding-1]))})
      new_dir_grid = tf.tidy(() => {return tf.variable(tf.zeros(new_grid.shape).cast('int32'))})}
    tf.dispose([grid,dirGrid])
    grid = tf.variable(new_grid.clone()); dirGrid = tf.variable(new_dir_grid.clone())
    tf.dispose([new_grid,new_dir_grid])
    scene.remove(mesh)
    mesh.dispose()
    generateVoxelgrid()
    updateVoxelGrid()
    scene.add(mesh)
    grid_size = new_grid_size   
    generate_kernel(radius, grid_size)
    }gridsizecounter.textContent = grid_size
    if (prevstate) {startstate()};
  }
function updateradius() {
  radiuscounter.textContent = radius.toFixed(0);
}

function updatet() {
  timecounter.textContent = T.toFixed(0);
}

function updatem() {  
  gammacounter.textContent = m.toFixed(2);
}

function updates() {
  mucounter.textContent = s.toFixed(2);
}
startstop.addEventListener("click", () => {
  updatestate()
  if (isRunning) {
    startstop.style.backgroundImage = "url('pausebutton.png')";
  } else {
    startstop.style.backgroundImage = "url('startbutton.png')";;
  }
});

function inc(value, increments,dir) {
  const range = increments[1] - increments[0]; // Calculate the range
  const step = dir*range / increments[2]; // Calculate the step size
  const nextValue = value + step; // Calculate the next value
  
  // Check if the next value exceeds the maximum
  if (nextValue > increments[1] || nextValue < increments[0]) {
  return value;
}
  
  return nextValue;
}