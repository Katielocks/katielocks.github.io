
tf.setBackend('webgl')
function dir_kernel(grid) {
 // Define the weights of the directional kernel
  let boolgrid = tf.notEqual(grid,0).cast('int32')
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
 return  dirgrid.clipByValue(0,1).cast('int32');
}
function generate_cell(local,value){
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshBasicMaterial({ color: new THREE.Color(0, value, 0) });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(local[0],local[1],local[2]);
  scene.add(mesh);
  return mesh
}
function generateVoxel(grid) {
  // Apply a directional kernel to the input grid, and extract the indices and values of the non-zero elements
  const dirGrid = dir_kernel(grid)
  let dirIndices = nonzero(dirGrid)
  const keys = dirIndices.arraySync().map(key => key.join(' '))
  const values = tf.gatherND(grid, dirIndices).arraySync();
  dirIndices = tf.sub(dirIndices,grid_size/2-0.5).arraySync()
  for (let i = 0; i < dirIndices.length; i++) {
    const key = keys[i]
    const mesh = generate_cell(dirIndices[i],values[i])
    Mesh_dict.set(key,mesh)
  }
  // Convert the values into a binary representation of the direction in which the voxel should be extruded
  return dirGrid;
}


function updateVoxel(new_grid,dirGrid,Mesh_dict) {
    let new_dir_grid = dir_kernel(new_grid)

    const diffgrid = tf.sub(new_dir_grid,dirGrid)
    // Cells to be deleted
    const deldirIndices = nonzero(tf.equal(diffgrid,-1)).arraySync().map(key => key.join(' '))  
    // Cells to be who face array needs to be Updated
    // Cells Generated
    let newdirIndices = nonzero(tf.equal(diffgrid,1))
    const newcellvalues = tf.gatherND(new_grid,newdirIndices).arraySync()
    const newkey = newdirIndices.arraySync().map(key => key.join(' '))
    newdirIndices = tf.sub(newdirIndices,grid_size/2-0.5).arraySync()
    // Deletes Cells
    for (let index = 0; index < deldirIndices.length; index++) {
    const local = deldirIndices[index]
    let mesh = Mesh_dict.get(local)
    scene.remove(mesh)
    mesh.geometry.dispose();
    mesh.material.dispose();
    mesh = null;
    Mesh_dict.delete(local)
  }
    // Get updated values from new_grid
    let updateInds = Array.from(Mesh_dict.keys());
    updateInds = updateInds.map(key => key.split(' ').map(Number))
    updateInds = tf.tensor(updateInds,[updateInds.length,3],dtype='int32')
    updatevalues = tf.gatherND(new_grid,updateInds).arraySync()
    // Apply updated values
    let index = 0;
    for (const [, mesh] of Mesh_dict.entries()) {
      mesh.material.color.setRGB(0, updatevalues[index], 0);
      index++;
    }
    for (let index = 0; index < newdirIndices.length; index++) {
      const Mesh =generate_cell(newdirIndices[index],newcellvalues[index])
      Mesh_dict.set(newkey[index],Mesh)
    }
    return new_dir_grid
  }

function transposecomplex(tensor,perms){
    const size = perms.map(index => tensor.shape[index])
    let real = tf.real(tensor).reshape(size).transpose(perms)
    let imag = tf.imag(tensor).reshape(size).transpose(perms)
    return tf.complex(real,imag)}

function fft1(tensor,perms){
let fttensor = tf.spectral.fft(tensor.transpose(perms))
return transposecomplex(fttensor,perms)}

function fftn(tensor){
let perms = tf.range(0,tensor.shape.length,1).arraySync()
let fttensor = fft1(tensor,perms)
for (let i=1;i<perms.length;i++){
const size = perms.length-1
const permsi = perms.slice();
[permsi[size],permsi[size-i]] = [permsi[size-i],permsi[size]];
fttensor = fft1(fttensor,permsi)
}
return fttensor}
function ifft1(tensor,perms){
let fttensor = tf.spectral.ifft(tensor.transpose(perms))
return transposecomplex(fttensor,perms)}

function ifftn(tensor){
let perms = tf.range(0,tensor.shape.length,1).arraySync()
let fttensor = ifft1(tensor,perms)
for (let i=1;i<perms.length;i++){
const size = perms.length-1
const permsi = perms.slice();
[permsi[size],permsi[i-1]] = [permsi[i-1],permsi[size]];
fttensor = ifft1(fttensor,permsi)
}
return fttensor}
function generate_kernel(radius, grid_size) {
  const mid = Math.floor(grid_size/2);
  const bell = (x, m, s) => tf.exp(tf.neg(tf.pow(tf.div(tf.sub(x, m), s), 2)).div(2));
  const x = tf.range(-mid, mid,1);
  let [X, Y] = tf.meshgrid(x, x);
  X = X.expandDims(0).tile([X.shape[0],1, 1]);
  Y = Y.expandDims(0).tile([Y.shape[0],1, 1])
  const Z = X.transpose();
  const D = tf.div(tf.sqrt(tf.add(tf.add(tf.pow(X, 2), tf.pow(Y, 2)), tf.pow(Z, 2))), radius);
  const K = tf.mul(tf.cast(tf.less(D, 1), 'float32'), bell(D, 0.5, 0.15));
  let fK = fftn(roll(K.div(tf.sum(K)), [0,1,2], [mid,mid,mid]).cast('complex64'))
  return fK
  }
  
function roll(grid, axis, shift) {
  const grid_size = tf.tensor(grid.shape).gather(axis).dataSync();
  const limits = [grid_size[0]-shift[0], grid_size[1]-shift[1], grid_size[2]-shift[2]];
  const [a1, b1] = tf.split(grid, [limits[0], shift[0]], axis[0]);
  const x1 = tf.concat([b1,a1],axis[0]);
  const [a2, b2] = tf.split(x1, [limits[1], shift[1]], axis[1]);
  const x2 = tf.concat([b2,a2],axis[1])
  const [a3, b3] = tf.split(x2, [limits[2], shift[2]], axis[2]);
  const x3 = tf.concat([b3,a3],axis[2])
  return x3;

  }
function golupdate(grid, kernel, frames_num, m, s) {
  const U = tf.real(ifftn(tf.mul(kernel,fftn(grid.cast('complex64'))))).reshape(grid.shape)
    return tf.add(grid,tf.div(growth(U, m, s), frames_num)).clipByValue(0, 1)
  }
  
  function growth(U, m, s) {
    return tf.sub(tf.mul(tf.exp(tf.sub(0,tf.div(U.sub(m).div(s).pow(2),2))),2),1)
  }
  

  function nonzero(t) {
    let values = tf.notEqual(t.flatten(),0).dataSync()
    let indices = Array.from(values.keys()).filter(i => values[i] !== 0)
    indices = tf.tensor(indices,shape =[indices.length,1] ,dtype='int32');
    let dim = indices 
    for (let index = 0; index < t.shape.length-1; index++) {
      const element = t.shape[index+1];
      dim = tf.floorDiv(dim,element)
      indices = tf.concat([dim,indices],axis=1)
    }
    indices = tf.mod(indices, tf.tensor(t.shape))
    return indices.cast('int32')
  
  }
    


  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
  const renderer = new THREE.WebGLRenderer();
  renderer.setSize( window.innerWidth, window.innerHeight );
  document.body.appendChild( renderer.domElement );


const grid_size = 30
const cluster_size = 10
      const m = 0.15
      const s = 0.07
      const frames_num = 10
      const radius = 5
const mid = Math.floor((grid_size -cluster_size)/ 2)
let grid = tf.pad(tf.ones([cluster_size,cluster_size,cluster_size]),[[mid,mid],[mid,mid],[mid,mid]]).cast('float32')
let Mesh_dict = new Map()
let dirGrid = generateVoxel(grid)
const local = tf.tensor([0,0,0]).cast('int32')
camera.position.x = 2*cluster_size ;
camera.position.y = 2*cluster_size ;
camera.position.z = 2*cluster_size ;
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; 
controls.dampingFactor = 0.05; // Set the damping factor for the damping effect
camera.position.z = 5;
const animate = function () {
  requestAnimationFrame( animate );
  controls.update(); // Update controls
  renderer.render( scene, camera );
};
animate();
kernel = generate_kernel(radius, grid_size)
let intervalId = null; // Variable to hold the interval ID

// Function to start the interval
function startInterval() {
  intervalId = setInterval(() => {
    new_grid = golupdate(grid, kernel, frames_num, m, s);
    dirGrid = updateVoxel(new_grid, dirGrid, Mesh_dict);
    grid = new_grid;
  }, 2000);
}

// Function to stop the interval
function stopInterval() {
  clearInterval(intervalId);
}

// Event listener for keydown event
document.addEventListener("keydown", (event) => {
  if (event.code === "Space") {
    if (intervalId) {
      // If interval is running, stop it
      stopInterval();
    } else {
      // If interval is not running, start it
      startInterval();
    }
  }
});
