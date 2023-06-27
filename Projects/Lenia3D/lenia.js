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

function generateseed() {
  seed = Math.floor(Math.random() * Math.pow(10, 8));
}

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
async function  resetVoxelGrid() {
  generategrid()
  updateVoxelGrid()
}

DIM = 3
DIM_DELIM = {0:'', 1:'$', 2:'%', 3:'#', 4:'@A', 5:'@B', 6:'@C', 7:'@D', 8:'@E', 9:'@F'}

function rle2arr(st) {
  var stacks = [];
  for (var dim = 0; dim < DIM; dim++) {
      stacks.push([]);
  }
  var last = '';
  var count = '';
  var delims = Object.values(DIM_DELIM);
  st = st.replace(/!$/, '') + DIM_DELIM[DIM - 1];
  for (var i = 0; i < st.length; i++) {
      var ch = st[i];
      if (/\d/.test(ch)) {
          count += ch;
      } else if ('pqrstuvwxy@'.includes(ch)) {
          last = ch;
      } else {
          if (!delims.includes(last + ch)) {
              _append_stack(stacks[0], ch2val(last + ch) / 255, count, true);
          } else {
              var dim = delims.indexOf(last + ch);
              for (var d = 0; d < dim; d++) {
                  _append_stack(stacks[d + 1], stacks[d], count, false);
                  stacks[d] = [];
              }
          }
          last = '';
          count = '';
      }
  }
  var A = stacks[DIM - 1];
  var max_lens = [];
  for (var dim = 0; dim < DIM; dim++) {
      max_lens.push(0);
  }
  _recur_get_max_lens(0, A, max_lens);
  _recur_cubify(0, A, max_lens);
  return tf.tensor(A);
}

function _append_stack(list1, list2, count, is_repeat = false) {
  list1.push(list2);
  if (count !== '') {
      var repeated = is_repeat ? list2 : [];
      for (var i = 0; i < parseInt(count) - 1; i++) {
          list1.push(repeated);
      }
  }
}

function ch2val(c) {
  if (c === '.' || c === 'b') return 0;
  else if (c === 'o') return 255;
  else if (c.length === 1) return c.charCodeAt(0) - 'A'.charCodeAt(0) + 1;
  else return (c.charCodeAt(0) - 'p'.charCodeAt(0)) * 24 + (c.charCodeAt(1) - 'A'.charCodeAt(0) + 25);
}

function _recur_get_max_lens(dim, list1, max_lens) {
  max_lens[dim] = Math.max(max_lens[dim], list1.length);
  if (dim < DIM - 1) {
      for (var i = 0; i < list1.length; i++) {
          _recur_get_max_lens(dim + 1, list1[i], max_lens);
      }
  }
}

function _recur_cubify(dim, list1, max_lens) {
  var more = max_lens[dim] - list1.length;
  if (dim < DIM - 1) {
      for (var i = 0; i < more; i++) {
          list1.push([]);
      }
      for (var i = 0; i < list1.length; i++) {
          _recur_cubify(dim + 1, list1[i], max_lens);
      }
  } else {
      for (var i = 0; i < more; i++) {
          list1.push(0);
      }
  }
}

async function  updateVoxelGrid() {
    let new_dir_grid = tf.tidy(() => {return dir_kernel(grid).cast('float32')})
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


const kernel_core = {
  0: (r) => tf.tidy(() => {return tf.pow(tf.mul(tf.mul(4,r),tf.sub(1, r)), 4)}),  // polynomial (quad4)
  1: (r) => tf.tidy(() => {return tf.exp(tf.sub(4,tf.div(1, tf.mul(r, tf.sub(1,r)))))}),  // exponential / gaussian bump (bump4)
  2: (r, q = 1 / 4) => tf.tidy(() => {return tf.logicalAnd(tf.greaterEqual(r, q), tf.lessEqual(r, 1 - q)).cast('float32')}),  // step (stpz1/4)
  3: (r, q = 1 / 4) => tf.tidy(() => {return tf.add(tf.logicalAnd(tf.greaterEqual(r, q), tf.lessEqual(r, 1 - q)).cast('float32'), tf.logicalAnd(tf.less(r, q), 0.5))  // staircase (life)
})};

const growth_func = {
0: (n, m,s) => tf.tidy(() => {return tf.sub(tf.mul(tf.pow(tf.maximum(0,tf.sub(1,tf.div(tf.pow(tf.sub(n,m),2),tf.mul(9,tf.pow(s,2))))),4),2),1) }),  // polynomial (quad4)
1: (n, m,s) => tf.tidy(() => {return tf.sub(tf.mul(tf.exp(tf.div(tf.neg(tf.pow(tf.sub(n, m), 2)), tf.mul(2, tf.pow(s, 2))))), 1)}),  // exponential / gaussian (gaus)
2: (n, m,s) => tf.tidy(() => {return tf.sub(tf.logicalAnd(tf.lessEqual(tf.abs(tf.sub(n, m)), s), 2), 1)})  // step (stpz)
};

function generate_kernel(){
  let dummy = tf.tidy(() => {
      const mid = Math.floor(SIZE/2);
      const x = tf.range(-mid, mid,1);
      let [X, Y] = tf.meshgrid(x, x);
      X = X.expandDims(0).tile([X.shape[0],1, 1]);
      Y = Y.expandDims(0).tile([Y.shape[0],1, 1])
      const Z = X.transpose();
      const D = tf.div(tf.sqrt(tf.add(tf.add(tf.pow(X, 2), tf.pow(Y, 2)), tf.pow(Z, 2))), worldparams['R'])
      let K = kernel_shell(D);
      K = roll(K.div(K.sum()), [0,1,2], [mid,mid,mid])
      KFFT = fftn(K.cast('complex64'));
      return KFFT
  })
  if (kernel.shape !== dummy.shape) {kernel.dispose()
  kernel = dummy}
  else{kernel.assign(dummy)
    dummy.dispose()}
  };
  function kernel_shell(r){
      return tf.tidy(() => {
      const B = worldparams['b'].length
      const Br = tf.mul(B,r);
      const bs  = worldparams['b']
      const b = map(bs,tf.minimum(tf.floor(Br).cast('int32'),B-1))

      const kfunc = kernel_core[worldparams['kn'] - 1]
      return tf.mul(tf.mul(tf.less(r,1),kfunc(tf.minimum(tf.mod(Br,1),1))),b)})
  }
  
function map(arr,tensor){
  const tensorarr = tensor.dataSync()
  const mappedarr  = tensorarr.map(x=>arr[x])
  tensor.dispose()
  return tf.tensor(mappedarr, tensor.shape);
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
    const fftngrid = fftn(grid.cast('complex64'))
    const potential_FFT = tf.mul(fftngrid, kernel)
    const dt = 1/worldparams['T']
    const mid = Math.floor(SIZE/2)
    const U = tf.real(ifftn(potential_FFT))
    gfunc = growth_func[worldparams['gn'] - 1]
    const field = gfunc(U, worldparams['m'], worldparams['s'])
    const grid_new = tf.add(grid,tf.mul(dt,field))
    return grid_new.clipByValue(0,1)
    })
    grid.assign(dummy)
    dummy.dispose()
    }
  function growth(U, m, s) {
    return tf.tidy(() => { return tf.sub(tf.mul(tf.exp(tf.sub(0,tf.div(U.sub(m).div(s).pow(2),2))),2),1)})
  }
  
  function initGrid(){
    grid = tf.tidy(() => {return tf.variable(tf.zeros([SIZE,SIZE,SIZE]).cast('float32'))})
  }
  function initdirGrid(){
    dirGrid = tf.tidy(() => {return tf.variable(tf.zeros(grid.shape).cast('float32'))})
  }
  function initKernel(){
    kernel = tf.tidy(() => {return tf.variable(tf.zeros([SIZE,SIZE,SIZE]).cast('complex64'))})
  }

  function generategrid(){
    
    if (seed == null){generateseed()}
    const mid = Math.floor((SIZE -cluster_size)/ 2)
    new_grid = tf.tidy(() => {return tf.variable(tf.pad(tf.randomUniform([cluster_size,cluster_size,cluster_size], 0, 1, 'float32', seed),[[mid,mid],[mid,mid],[mid,mid]]).cast('float32'))})

    grid.assign(new_grid)
    new_grid.dispose()}
let SIZE = 64
let cluster_size = 10
seed = null
let worldparams = {'R':15, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'kn':1, 'gn':1}
let grid = null, dirGrid = null, kernel = null, mesh = null, offset = null, cubeProxy = new THREE.Object3D()
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

camera.position.x = 1.2*SIZE ;
camera.position.y = 1.2*SIZE ;
camera.position.z = 1.2*SIZE ;
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; 
controls.dampingFactor = 0.05; // Set the damping factor for the damping effect
scene.background = new THREE.Color(0x101214);
let isRunning = false; // Variable to hold the interval ID


function animatestate() {
  if (isRunning) {
    // Update the grid every 1/T seconds
    setTimeout(function() {
        golupdate();
        updateVoxelGrid()
        animatestate();
        console.log(tf.memory().numTensors)
      }, Math.floor(1000/worldparams['T']))

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
    startstop.style.backgroundImage = "url('startbutton.png')";
    stopstate();
  } else {
    startstop.style.backgroundImage = "url('pausebutton.png')";
    startstate();
  }
}

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

// Loop through each dictionary in 'animalsarr' and create menu items



// Get the DOM elements for display








const gridsizecounter = document.getElementById("gridsize");
const radiuscounter = document.getElementById("radius");
const timecounter = document.getElementById("time");
const mucounter = document.getElementById("mu");
const sigmacounter = document.getElementById("sigma");
const seedinput = document.getElementById("seedinput");
const seedbutton = document.getElementById("seed-button");
const seedcontainer = document.getElementById("seedcont");
const seeddisplay = document.getElementById("seed");
const namecontainer = document.getElementById("namecont");;
const namedisplay = document.getElementById("name");
// Get the buttons for adding and subtracting
const gridsizeslider = document.getElementById("gridsize-slider");
const radiusAddButton = document.getElementById("radius-add-button");
const radiusSubButton = document.getElementById("radius-sub-button");
const timeAddButton = document.getElementById("time-add-button");
const timeSubButton = document.getElementById("time-sub-button");
const muslider = document.getElementById("mu-slider");
const sigmaslider = document.getElementById("sigma-slider");
const startstop = document.getElementById("startstop");
const reset = document.getElementById("reset");
let new_grid_size = SIZE
let gridrange = [2,100,49]
let rrange = [0,100,100]
let trange = [0,20,20]
updategridsize();
updateradius();
updatet();
updatem();
updates();

gridsizeslider.value = SIZE
muslider.value = worldparams['m']
sigmaslider.value = worldparams['s']
// Add event listeners to the buttons
gridsizeslider.addEventListener("change", () => {
  new_grid_size = gridsizeslider.value
  updategridsize();
});
radiusAddButton.addEventListener("click", () => {
  worldparams['R'] = inc(worldparams['R'], rrange,1)
  updateradius();
});

radiusSubButton.addEventListener("click", () => {
  worldparams['R'] = inc(worldparams['R'], rrange,-1)
  updateradius();
});

timeAddButton.addEventListener("click", () => {
  worldparams['T'] = inc(worldparams['T'], trange,1)
  updatet();
});

timeSubButton.addEventListener("click", () => {
  worldparams['T'] = inc(worldparams['T'], trange,-1)
  updatet();
});

muslider.addEventListener("change", () => {
  worldparams['m']  = parseFloat(muslider.value) 
  updatem();
});

sigmaslider.addEventListener("change", () => {
  worldparams['s']  = parseFloat(sigmaslider.value)
  updates();
});



reset.addEventListener("click", () => {
  resetVoxelGrid();
});
  function updategridsize() {
    let padding  = (new_grid_size-SIZE)/2
    prevstate = isRunning
    if (padding != 0) { 
      stopstate()
    if (padding > 0) {

      new_grid = tf.tidy(() => {return tf.variable(tf.pad(grid,[[padding,padding],[padding,padding],[padding,padding]]))})
      new_dir_grid = tf.tidy(() => {return tf.variable(tf.zeros(new_grid.shape).cast('float32'))})}
    else { 
      padding = -padding
      new_grid = tf.tidy(()=>{return tf.variable(grid.slice([padding,padding,padding],[SIZE-padding-1,SIZE-padding-1,SIZE-padding-1]))})
      new_dir_grid = tf.tidy(() => {return tf.variable(tf.zeros(new_grid.shape).cast('float32'))})}
    tf.dispose([grid,dirGrid])
    grid = tf.variable(new_grid.clone()); dirGrid = tf.variable(new_dir_grid.clone())
    tf.dispose([new_grid,new_dir_grid])
    scene.remove(mesh)
    mesh.dispose()
    generateVoxelgrid()
    updateVoxelGrid()
    scene.add(mesh)
    SIZE = new_grid_size   
    generate_kernel()
    }gridsizecounter.textContent = SIZE
    if (prevstate) {startstate()};
  }
function updateradius() {
  radiuscounter.textContent = worldparams['R'].toFixed(0);
}

function updatet() {
  timecounter.textContent = worldparams['T'].toFixed(0);
}

function updatem() {  
  mucounter.textContent = worldparams['m']
}

function updates() {
  sigmacounter.textContent = worldparams['s']
}
startstop.addEventListener("click", () => {
  updatestate()
});

function updateworldparams() {
  updateradius();
  updatet();
  updatem();
  updates();
}

function validateSeedInput(input) {
  const seedLength = input.value.length;
  // Remove any non-numeric characters from the input value
  input.value = input.value.replace(/\D/g, '');
  input.value = input.value.slice(0, 8);
  if (seedLength < 8 && seedLength > 0) {
    seedWarning.style.visibility = 'visible';
  } else {
    seedWarning.style.visibility = 'hidden';
  }
}




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



//  GENERATION

seedbutton.addEventListener("click", () => {
  seed = parseInt(seedinput.value);
  resetVoxelGrid();
  updateseed();
})
function updateseed() {
  if (window.getComputedStyle(seedcontainer).display === "none") {
    seedcontainer.style.display = "block";
  }
  if (window.getComputedStyle(namecontainer).display === "block") {
    namecontainer.style.display = "none";
  }
  seeddisplay.textContent = seed;
}

function updatename(str) {
  
  
  if (window.getComputedStyle(seedcontainer).display === "block") {
    seedcontainer.style.display = "none";
  }
  if (window.getComputedStyle(namecontainer).display === "none") {
    namecontainer.style.display = "block";
  }
  namedisplay.textContent = str;
}
function PopulateAnimalList() {
  
  const fuck = animalArr[4]
	if (!animalArr) return;
	var list = document.getElementById("AnimalList");
	if (!list) return;
	list.innerHTML = "";
	var lastCode = "";
	var lastEng0 = "";
	var lastChi0 = "";
	var node = list;
	var currLevel = 0;
	for (var i=0; i<animalArr.length; i++) {
		var a = animalArr[i];
		if (Object.keys(a).length >= 3) {
			var codeSt = a['code'].split("(")[0];
			var engSt = a['name'].split(" ");
      
			if (codeSt.startsWith("~")) codeSt = codeSt.substring(1);
			if (codeSt.startsWith("*")) codeSt = "";
			var sameCode = codeSt != "" && (codeSt == lastCode);
			var sameEng0 = engSt[0] != "" && (engSt[0] == lastEng0);
			lastCode = codeSt;
			lastEng0 = engSt[0];
			if (Object.keys(a).length >= 4) {
				var cellSt = a['cells'];
				var st = cellSt.split(";");
				var ruleSt = "";
				if (st.length >= 3) ruleSt = st[0].trim()+";"+st[1].trim()+";"+st[2].trim();
				var li = node.appendChild(document.createElement("LI"));
				li.classList.add("action");
				li.title = a[0] + " " + engSt.join(" ") + "\n" + ruleSt;
				if (sameCode) codeSt = "âˆ’".repeat(codeSt.length);
				if (sameEng0) engSt[0] = lastEng0.substring(0, 1) + ".";
				li.innerHTML = codeSt + " " + engSt.join(" ")  + " " + GetLayerSt(ruleSt);
				li.dataset["animalid"] = i;
				li.addEventListener("click", SelectAnimalItem);
			} else if (Object.keys(a).length == 3) {
				var nextLevel = parseInt(codeSt.substring(1));
				var diffLevel = nextLevel - currLevel;
				var backNum = (diffLevel<=0) ? -diffLevel+1 : 0;
				var foreNum = (diffLevel>0) ? diffLevel : 1;
				for (var k=0; k<backNum; k++) {
					node = node.parentElement;
					if (node.tagName == "LI") node = node.parentElement;
				}
				node = node.appendChild(document.createElement("LI"));
				node.classList.add("group");
				var div = node.appendChild(document.createElement("DIV"));
				div.title = engSt;
				div.innerHTML = engSt;
				div.addEventListener("click", function (e) { this.parentElement.classList.toggle("closed"); });
				for (var k=0; k<foreNum; k++) {
					node = node.appendChild(document.createElement("UL"));
				}
				currLevel = nextLevel;
			}
		}
	}
}
function GetLayerSt(st) {
	if (!st) return "";
	var s = st.split(/\=|\;|\*/);
	var s3 = s[3].split(/\(|\)|\,/);
	var l = Math.max(s3.length - 3, 0);
	return "[" + (l+1) + "]";
}
function SelectAnimalItem(e) {
  const rundummy = isRunning
  if (rundummy) {updatestate()}
	var item = e.target;
	if (!item.dataset["animalid"]) item = item.parentElement;
	var id = parseInt(item.dataset["animalid"]);
	SelectAnimalID(id);
  if (rundummy) {updatestate()}
}

function SelectAnimalID(id) {
  console.log(id)
  if (id < 0 || id >= animalArr.length) return;
  var a = animalArr[id];
  if (Object.keys(a).length < 4) return;
  const cellSt = a['cells'];
  let dummy = tf.tidy(() =>{ 
    let newgrid  = rle2arr(cellSt);
    let mid = newgrid.shape.map(x=>SIZE-x)
    const realign = mid.map(x=>x%2)
    mid = mid.map(x=>Math.floor(x/2))
    newgrid = tf.pad(newgrid, [[mid[0], mid[0]], [mid[1], mid[1]], [mid[2], mid[2]]])
    newgrid = tf.pad(newgrid, [[realign[0], 0], [realign[1], 0], [realign[2], 0]])
    return newgrid
  });
  grid.assign(dummy);
  worldparams = a['params']
  worldparams['b'] = bparamfix(worldparams['b'])
  dummy.dispose();
  updateVoxelGrid();
  updateworldparams()
  generate_kernel();
  updatename(a['name'])
}


PopulateAnimalList()


function bparamfix(string) {
  let splitValues = string.split(","); // Split the string by commas

  let result = [];

  for (let i = 0; i < splitValues.length; i++) {
    let value = splitValues[i];

    if (value.includes("/")) {
      let fractionParts = value.split("/");
      let numerator = parseInt(fractionParts[0]);
      let denominator = parseInt(fractionParts[1]);
      let convertedValue = numerator / denominator;
      result.push(convertedValue);
    } else {
      result.push(parseInt(value));
    }
  }

  return result;
}

function zoom3D(Tensor,zoom){
  return tf.tidy(()=>{
  const shape = Tensor.shape
  const zoomedshape = shape.slice(1).map(x => Math.round(x*zoom))
  const index = tf.range(0,shape[0],1).reshape([1,1,shape[0],1])
  Tensor = Tensor.reshape([shape[0],1,...shape.slice(1),1])
  let zoomedindex = tf.image.resizeNearestNeighbor(index,[1,Math.round(zoom*shape[0])],alignCorners =true)
  zoomedindex = zoomedindex.flatten().arraySync()
  let output = []
  for (let i=0;i<zoomedindex.length;i++){
   	let zoomed2d = tf.image.resizeNearestNeighbor(Tensor.gather(zoomedindex[i]),zoomedshape,alignCorners =true)
    zoomed2d = zoomed2d.reshape(zoomedshape)
    output.push(zoomed2d)
  }
  output = tf.stack(output)
  return output})}

  initGrid()
initdirGrid()
initKernel()
generateVoxelgrid()
if (animalArr !== null) {
  SelectAnimalID(40);
} else {
  generateGrid();
  generateKernel();
  updateVoxelGrid();
}
const animate = function () {
  requestAnimationFrame( animate );
  controls.update(); // Update controls
  renderer.render( scene, camera );
};
  animate();
