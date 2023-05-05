
function dir_kernel(grid) {

  /**
  * Use a 3D edge detection kernel to derive number and position of voxel faces.
 * @param {tf.Tensor} grid - A 3D TensorFlow.js tensor.
 * @returns {tf.Tensor} dir_grid - Filtered TensorFlow.js tensor with values representing the number and direction of voxel faces.
 */

 // Convert the grid to a boolean tensor and add two dimensions to make it compatible with the convolution operation
 grid = tf.reshape(grid,[1,grid.shape[0],grid.shape[1],grid.shape[2],1]).cast('bool')

 // Define the weights of the directional kernel
 const kern_weights = tf.tensor([
        [[ 0, 0, 0],
        [ 0, -4, 0],
        [ 0, 0, 0]],
       [[    0,   -32,     0],
        [   -1, 63,   -8],
        [    0,   -16,     0]],

       [[ 0,     0, 0],
        [    0,    -2,     0],
        [ 0,     0, 0]]], [3, 3, 3], 'int32');
 const kernel = tf.reshape(kern_weights, [3, 3, 3, 1, 1])
 const dir_grid =  tf.clipByValue(tf.conv3d(grid, kernel, [1, 1, 1,1,1], 'same'), 0, Infinity).cast('int32');
 // Remove the extra dimensions and return the edge detected grid
 return tf.reshape(dir_grid,[grid.shape[1],grid.shape[2],grid.shape[3]]);
}

function generateVoxel(grid) {
  /**
   * Generate cuboid voxel mesh of a given 3D grid.
   * @param {Array} grid - A 3D array representing the input grid.
   * @returns {Object} A dictionary containing:
   * - vertices: A 2D array of shape (N, 3) representing the unique vertices of the voxel geometry.
   * - faces: A 2D array of shape (M, 4) representing the faces of the voxel geometry, where each row is a set of indices into the vertices tensor.
   * - face_values: A 1D array of length M representing the value of each face of the voxel geometry, taken from the corresponding index in the input grid.
   */

  // Define the eight vertices of a cube with an offset tensor
  const offset = tf.tensor([
    [0, 0, 0], // vertex 0
    [1, 0, 0], // vertex 1
    [1, 1, 0], // vertex 2
    [0, 1, 0], // vertex 3
    [0, 0, 1], // vertex 4
    [1, 0, 1], // vertex 5
    [1, 1, 1], // vertex 6
    [0, 1, 1], // vertex 7
  ]);

  // Define the faces of the cube as offsets into the vertex tensor
  const offsetFaces = tf.tensor([
    [0, 1, 2, 3], // face 0
    [1, 5, 6, 2], // face 1
    [4, 0, 3, 7], // face 2
    [5, 4, 7, 6], // face 3
    [3, 2, 6, 7], // face 4
    [4, 5, 1, 0], // face 5
  ]);

  // Apply a directional kernel to the input grid, and extract the indices and values of the non-zero elements
  const dirGrid = dir_kernel(grid);
  const dirIndices = nonzero(dirGrid)
  const dirValues = tf.gatherND(dirGrid, dirIndices);
  // Convert the values into a binary representation of the direction in which the voxel should be extruded
  let direction = tf.mod(tf.floorDiv(dirValues.expandDims(1), tf.pow(2, tf.range(5, -1, -1))), 2);
  direction = tf.reverse(direction, [1]);
  const face_indices  =  nonzero(direction);
  let faces = tf.gather(offsetFaces, face_indices.slice([0,1],[face_indices.shape[0], 1]).flatten()).cast('int32') ;
  const vertices_indices = face_indices.slice([0,0],[face_indices.shape[0], 1])
  const vertices = offset.gather(faces.flatten()).add(tf.gather(dirIndices, vertices_indices.tile([1,4]).flatten())).cast('int32');

  let verticesGrid = tf.zeros([grid.shape[0] + 1, grid.shape[1] + 1, grid.shape[2] + 1], 'int32');
  verticesGrid = tf.tensorScatterUpdate(verticesGrid,vertices, tf.ones([vertices.shape[0]], 'int32'));
  const uniqueVertices = nonzero(verticesGrid);
  verticesGrid = tf.tensorScatterUpdate(verticesGrid,uniqueVertices,  tf.range(0,uniqueVertices.shape[0],1, 'int32'));
  tf.gatherND(verticesGrid,vertices )
  faces = tf.reshape(tf.gatherND(verticesGrid,vertices ), [faces.shape[0], 4])
  const values = tf.gatherND(grid,dirIndices)
  const face_values = values.gather(vertices_indices).tile([1,18]).flatten()
  const face_colours = face_values.expandDims(0).transpose().pad([[0, 0],[1,1]]).flatten()
  faces =  generateTriangles(faces)
  const vertices2 = tf.add(uniqueVertices,tf.mul(tf.div(tf.tensor(grid.shape),2).expandDims(0).tile([uniqueVertices.shape[0],1]),-1))
  return {
    vertices :vertices2,
    faces,
    face_colours
  };
}
function generateTriangles(shapes) {
  const [a,b,c,d] = tf.split(shapes,[1,1,1,1],1)
  const triangles = tf.concat([a,b,c,a,c,d],1).reshape([2*shapes.shape[0],3])

  return triangles;
}

function generate_kernel(radius, grid_size) {

  const mid = Math.floor(grid_size/2);
  const bell = (x, m, s) => tf.exp(tf.neg(tf.pow(tf.div(tf.sub(x, m), s), 2)).div(2));
  const x = tf.range(-mid, mid,1);
  const [X, Y] = tf.meshgrid(x, x);
  const D = tf.div(tf.sqrt(tf.add(tf.pow(X.expandDims(0).tile([grid_size,1, 1]), 2), tf.add(tf.pow(X.expandDims(0).tile([grid_size,1, 1]).transpose(), 2), tf.pow(Y.expandDims(0).tile([grid_size,1, 1]), 2)))), radius);
  const K = tf.mul(tf.cast(tf.less(D, 1), 'float32'), bell(D, 0.5, 0.15));
  const fK = tf.spectral.fft(roll(K.div(tf.sum(K)), [0,1,2], [grid_size/2,grid_size/2,grid_size/2]).cast('complex64'));
  return fK;
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
    const a  =tf.spectral.ifft(tf.mul(kernel, tf.spectral.fft(grid.cast('complex64'))))
    const U = tf.real(tf.spectral.ifft(tf.mul(kernel, tf.spectral.fft(grid.cast('complex64'))))).reshape(grid.shape);
    const A = tf.add(grid,tf.div(growth(U, m, s), frames_num)).clipByValue(0, 1)
    return A
  }
  
  function growth(U, m, s) {
    const bell = (x, m, s) => tf.sub(tf.exp(tf.neg(tf.pow(tf.div(tf.sub(x, m), s), 2)).div(2)), 0.5);
    return tf.sub(tf.pow(bell(U, m, s), 2), 1);
  }
  

  function nonzero(t) {
    let values = tf.notEqual(t.flatten(),0).dataSync()
    let indices = Array.from(values.keys()).filter(i => values[i] !== 0)
    indices = tf.tensor(indices,shape =[indices.length,1] ,dtype='int32');
    let dim = indices 
    for (let index = 0; index < t.shape.length-1; index++) {
      const element = t.shape[index];
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


const grid_size = 10
      const m = 0.15
      const s = 0.05
      const frames_num = 10
      const radius = 12
let grid = tf.randomUniform([grid_size,grid_size,grid_size],0,1,'float32')
console.log("grid gen") 
let mesh_attributes =  generateVoxel(grid)
console.log("mesh gen")
let geometry = new THREE.BufferGeometry();
geometry.setIndex(mesh_attributes.faces.flatten().arraySync());
geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(mesh_attributes.vertices.flatten().arraySync()), 3));
geometry.setAttribute( 'color', new THREE.Float32BufferAttribute(mesh_attributes.face_colours.arraySync(), 3 ) );
let material = new THREE.MeshBasicMaterial( { vertexColors: true } );
let mesh = new THREE.Mesh(geometry, material);
scene.add(mesh)

camera.position.x = 2*grid_size;
camera.position.y = 2*grid_size;
camera.position.z = 2*grid_size;

  const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; 
controls.dampingFactor = 0.05; // Set the damping factor for the damping effect
camera.position.z = 5;
const animate = function () {
  requestAnimationFrame( animate );
  controls.update(); // Update controls
  renderer.render( scene, camera );
};

const kernel = generate_kernel(radius, grid_size);

function update(up_grid) {;
  let up_mesh_attributes = generateVoxel(up_grid);
  const positions = new Float32Array(up_mesh_attributes.vertices.flatten().arraySync());
  const colors = new Float32Array(up_mesh_attributes.face_colours.arraySync());

  const geometry = mesh.geometry; // Get the existing geometry
  
  mesh.geometry.dispose();

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(up_mesh_attributes.faces.flatten().arraySync());


  geometry.attributes.position.needsUpdate = true;
  geometry.attributes.color.needsUpdate = true;
  geometry.index.needsUpdate = true;
  return up_grid;
}

animate();

setInterval(() => {
  grid = grid = golupdate(grid, kernel, frames_num, m, s)
  update(grid);
}, 5000);

