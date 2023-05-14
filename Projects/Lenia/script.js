const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );

const geometry = new THREE.BufferGeometry();

const vertices = new Float32Array( [
    0, 0, 0, 
    1, 0, 0,
    1, 1, 0, 
    0, 1, 0,
    0, 0, 1, 
    1, 0, 1, 
    1, 1, 1, 
    0, 1, 1
] );

const indices = [
	0, 1, 2,
    0, 2, 3,
    1, 5, 6,
    1, 6, 2,
    4, 0, 3,
    4, 3, 7,
    5, 4, 7,
    5, 7, 6,
    3, 2, 6,
    3, 6, 7,
    4, 5, 1,
    4, 1, 0

];

geometry.setIndex( indices );
geometry.setAttribute( 'position', new THREE.BufferAttribute( vertices, 3 ) );

const material = new THREE.MeshBasicMaterial( { color: 0xff0000 } );
const cube = new THREE.Mesh( geometry, material );
scene.add( cube );
camera.position.z = 5;

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; 
controls.dampingFactor = 0.05; // Set the damping factor for the damping effect

const animate = function () {
  requestAnimationFrame( animate );

  controls.update(); // Update controls

  renderer.render( scene, camera );
};

animate();