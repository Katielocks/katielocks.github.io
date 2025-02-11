import { drawTexture, syncWait,resetFramebuffer } from './gldraw.js';

// Pass a `WebGLData` object and specify a shape yourself.

// This makes it possible for TF.js applications to avoid GPU / CPU sync.
// For example, if your application includes a preprocessing step on the GPU,
// you could upload the GPU output directly to TF.js, rather than first
// downloading the values.

// Example for WebGL2:
const customCanvas = document.getElementById('canvas');
const customBackend = new tf.MathBackendWebGL(customCanvas);
tf.registerBackend('custom-webgl', () => customBackend);
const gl = customCanvas.getContext('webgl2');

    class Lenia {
        constructor() {
            this.params = {'R':15, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'kn':1, 'gn':1};
            this.DIM = 3;

            this.render = '3D'

            this.size = Array.from({ length: this.DIM }).fill(64);

            this.paramsbackup = { ...this.params };
            this.sizebackup = this.size.slice();

            this.cluster_size = this.size.map(x => Math.round(x*0.8));
            this.cluster_density = 1
            this.cluster_range = [0,1]

            this.colourbarmax = [253 / 255, 232 / 255, 64 / 255];
            this.colourbarmin = [68 / 255, 13 / 255, 84 / 255];


            this.seed = null
            this.id = null
            this.isRunning = false;
            this.time = 0;
            this.gen = 0;
            this.framesper = 1;
            this.framecounter = 0;
            this.UI = new UI(this);
            this.tensor = new tensor(this);
            this.mesh = new mesh(this);
            this.mesh.render    ();
            this.tensor.generateKernel();

            this.tensorarray = []

            this.UI.updateparams()

            this.UI.menumode = 'pregenerated';  

            this.UI.editMenu(this.UI.menumodes[this.UI.menumode])

            this.UI.loading.style.display = 'none'

            }
            animatestate() {
                if (this.isRunning) {
                // Update the grid every 1/T seconds
                setTimeout(function() {
                    this.tensor.update()
                    this.gen += 1
                    this.time += 1/this.params['T'] 
                    if (this.gen % this.framesper ==- 0) {this.mesh.render()};
                    this.UI.updatetime()
                    this.UI.updategen()
                    this.UI.updateframe()
                    this.animatestate();
                }.bind(this), Math.floor(1000 / this.params['T']));
            
                }
            }
            
            // Function to start the game
            startstate() {
                this.isRunning = true;
                this.animatestate();
            }

            // Function to stop the game
            stopstate() {
                this.isRunning = false;
            }
            updatestate() {
                if (this.isRunning) {
                    this.stopstate();
                } else {
                    this.startstate();
                }
            }
            reset() {
                this.resetparams();
                this.tensor.updateGridSize();
                this.UI.updateparams();
                this.renderReset();
                if (this.id !== null) {
                    this.tensor.load.SelectAnimalID(this.id)
                }
                else{
                    this.tensor.randomGrid()
                    this.renderUpdate()
                }
                this.time = 0;
                this.gen = 0;
                this.frame = 0;
            }
            resetparams() {
                this.params = { ...this.paramsbackup };
                this.size = this.sizebackup.slice();
            }
            setparams(){
                this.paramsbackup = { ...this.params };
            }
            colormap(value){
                return this.colourbarmax.map(
                    (x, i) => (x - this.colourbarmin[i]) * value + this.colourbarmin[i]
                );
            }
            



    }


    class tensor {
        constructor(lenia) {
        this.lenia = lenia;
        this.init(true);
        this.load = new Load(this.lenia);
        this.kernel_core = {
            0: (r) => tf.tidy(() => tf.pow(tf.mul(tf.mul(4, r), tf.sub(1, r)), 4)), // polynomial (quad4)
            1: (r) => tf.tidy(() => tf.exp(tf.sub(4, tf.div(1, tf.mul(r, tf.sub(1, r)))))), // exponential / gaussian bump (bump4)
            2: (r, q = 1 / 4) => tf.tidy(() => tf.logicalAnd(tf.greaterEqual(r, q), tf.lessEqual(r, 1 - q)).cast('float32')), // step (stpz1/4)
            3: (r, q = 1 / 4) => tf.tidy(() => tf.add(tf.logicalAnd(tf.greaterEqual(r, q), tf.lessEqual(r, 1 - q)).cast('float32'), tf.logicalAnd(tf.less(r, q), 0.5))), // staircase (life)
        };
        this.growth_func = {
            0: (n, m,s) => tf.tidy(() => {return tf.sub(tf.mul(tf.pow(tf.maximum(0,tf.sub(1,tf.div(tf.pow(tf.sub(n,m),2),tf.mul(9,tf.pow(s,2))))),4),2),1) }),  // polynomial (quad4)
            1: (n, m,s) => tf.tidy(() => {return tf.sub(tf.mul(tf.exp(tf.div(tf.neg(tf.pow(tf.sub(n, m), 2)), tf.mul(2, tf.pow(s, 2))))), 1)}),  // exponential / gaussian (gaus)
            2: (n, m,s) => tf.tidy(() => {return tf.sub(tf.logicalAnd(tf.lessEqual(tf.abs(tf.sub(n, m)), s), 2), 1)})  // step (stpz)
        };
        this.generateKernel();

        }
        init(init = false) {
            if (init){this.grid = tf.tidy(() =>  {return tf.variable(tf.zeros(this.lenia.size))});}
            this.gridVoxel = tf.tidy(() => {return tf.variable(tf.zeros(this.lenia.size.slice(-3)).cast('int32'))});
            this.kernel = tf.tidy(() => {return tf.variable(tf.zeros(this.lenia.size).cast('complex64'))});
            this.ekernel  = tf.tidy(() => {
                const size = this.lenia.size.slice(-3).map(x => x+2)
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
            const mid = size.map(x => Math.floor(x/2))
            const kernel = this.roll(this.resize(kern_weights,size),[0,1,2],mid)
            return this.fftn(kernel.cast('complex64'));
            });
        }
        transposeComplex(tensor, perm) {
            return tf.tidy(() => {
              const realPart = tf.real(tensor).reshape(tensor.shape).transpose(perm);
              const imagPart = tf.imag(tensor).reshape(tensor.shape).transpose(perm);
              return tf.complex(realPart, imagPart);
            });
          }
        
          fftAlongAxis(x, axis) {
            return tf.tidy(() => {
              const rank = x.shape.length;
              const perm = Array.from({ length: rank }, (_, i) => i);
              [perm[axis], perm[rank - 1]] = [perm[rank - 1], perm[axis]];
              console.log(perm)
              const transposed = this.transposeComplex(x, perm);
              const freq = tf.spectral.fft(transposed);
              return this.transposeComplex(freq, perm);
            });
          }
        
          fftn(x) {
            return tf.tidy(() => {
              const rank = x.shape.length;
              let out = x;
              for (let axis = 0; axis < rank; axis++) {
                out = this.fftAlongAxis(out, axis);
              }
              return out;
            });
          }
        
        ifftAlongAxis(x, axis) {
            return tf.tidy(() => {
              const rank = x.shape.length;
              const perm = Array.from({ length: rank }, (_, i) => i);
              [perm[axis], perm[rank - 1]] = [perm[rank - 1], perm[axis]];
          
              const transposed = this.transposeComplex(x, perm);
              const time = tf.spectral.ifft(transposed);
          
              return this.transposeComplex(time, perm);
            });
          }
        
        ifftn(x) {
            return tf.tidy(() => {
              const rank = x.shape.length;
              let out = x;
              for (let axis = 0; axis < rank; axis++) {
                out = this.ifftAlongAxis(out, axis);
              }
              return out;
            });
        }

        
          createCoordinateGrids(size) {
            const rank = size.length;
            const coords = [];
            for (let i = 0; i < rank; i++) {
              const mid = Math.floor(size[i] / 2), extra = size[i] % 2;
              let range = tf.range(-mid, mid + extra, 1);
              const shape = Array(rank).fill(1);
              shape[i] = size[i];
              range = range.reshape(shape);
              const tile = size.map((val, idx) => (idx === i ? 1 : val));
              coords.push(range.tile(tile));
            }
            return coords;
          }
        
          createDistanceField(size, R) {
            const coords = this.createCoordinateGrids(size);
            let D = tf.zeros(size);
            coords.forEach(coord => { D = D.add(coord.square()); });
            return D.sqrt().div(R);
          }
        
          kernelShell(r, params, kernelCoreFn) {
            return tf.tidy(() => {
              const bs = tf.tensor1d(params.b, 'float32');
              const B = bs.shape[0];
              const Br = r.mul(B);
              const floorBr = Br.floor();
              const idx = tf.minimum(floorBr, B - 1);
              const bVals = bs.gather(idx.cast('int32'));
              const fraction = Br.mod(1);
              const mask = r.less(1);
              const kfuncVals = kernelCoreFn(fraction);
              return mask.mul(kfuncVals).mul(bVals);
            });
          }
        
          generateKernel() {
            const { size, params } = this.lenia;
            const kfunc = this.kernel_core[params.kn - 1];
            const dummy = tf.tidy(() => {
              const D = this.createDistanceField(size, params.R);
              let K = this.kernelShell(D, params, kfunc);
              K = K.div(K.sum());
              const mid = size.map(x => Math.floor(x / 2));
              K = this.roll(K, [0,1,2], mid);
              K = K.cast('complex64');
              return this.fftn(K);
            });
            this.kernel.assign(dummy);
            dummy.dispose();
          }
        map(arr,tensor){
            const tensorarr = tensor.dataSync()
            const mappedarr  = tensorarr.map(x=>arr[x])
            tensor.dispose()
            return tf.tensor(mappedarr, tensor.shape);
            }
        roll(grid, axes, shifts) {
            return tf.tidy(() => {
                let result = grid;
                axes.forEach((axis, i) => {
                    const size = result.shape[axis];
                    const shift = shifts[i];
                    
                    // Calculate effective shift (handles negative and overshift values)
                    const effectiveShift = ((shift % size) + size) % size;
                    if (effectiveShift === 0) return;  // Skip if no actual shift
                    
                    // Split and concatenate
                    const splitPoint = size - effectiveShift;
                    const [a, b] = tf.split(result, [splitPoint, effectiveShift], axis);
                    result = tf.concat([b, a], axis);
                });
                return result;
            });
        }
        resize(grid,size){
            if (grid.dtype === 'complex64'){
                const real = tf.real(grid); const imag = tf.imag(grid)
                return tf.tidy(() => {return tf.complex(this.resize(real,this.lenia.size),this.resize(imag,this.lenia.size))})
            }
            let padding =  size.map((x,i)=>x-grid.shape[i])
            const error = padding.map(x=>Math.abs(x%2))
            padding = padding.map(x=>Math.floor(x/2))
            padding = padding.map((x,i)=>[x,x+error[i]])
            return tf.pad(grid,padding)
        }
        randomGrid(){
            const new_grid = tf.tidy(() => {
                const randomGrid = tf.randomUniform(this.lenia.cluster_size, this.lenia.cluster_range[0], this.lenia.cluster_range[1], 'float32', seed)
                const randomBoolGrid = tf.randomUniform(this.lenia.cluster_size, 0, 1, 'float32', seed).less(this.lenia.cluster_density)
                const new_grid = tf.mul(randomGrid,randomBoolGrid)
                return tf.cast(this.resize(new_grid,this.lenia.size),'float32')})
            this.grid.assign(new_grid)
            this.lenia.mesh.render()
            new_grid.dispose()
        }
        generateRandomGrid(){
            this.lenia.id = null;
            this.lenia.time = 0;
            this.lenia.gen = 0;
            this.randomGrid();
            this.lenia.mesh.render();
            /*
            this.lenia.UI.updateseed()*/
        }
        generateseed() {
            this.lenia.seed = Math.floor(Math.random() * Math.pow(10, 8));
        }
        edges() {
            // Define the weights of the directional kernel
            return tf.tidy(() => {
                const padding = [[1, 1], [1, 1], [1, 1]];
                const tolerance = 1e-18;
                const grid_bool = tf.greater(tf.pad(this.grid,padding), tolerance)
                let egrid = tf.real(this.ifftn(tf.mul(this.fftn(grid_bool.cast('complex64')), this.ekernel)))
                egrid = tf.round(egrid).slice(0,this.lenia.size)
            return  egrid.clipByValue(0,1).cast('int32')})
        }
        
        async update() {
            if (this.lenia.isRunning === false) {return}
            const complexgrid = this.grid.cast('complex64')
            const fftngrid = this.fftn(complexgrid)
            const potential_FFT = tf.mul(fftngrid, this.kernel)
            const dt = 1/this.lenia.params['T']
            const complexU = this.ifftn(potential_FFT)
            const U = tf.real(complexU)
            const gfunc = this.growth_func[this.lenia.params['gn'] - 1]
            const field = gfunc(U, this.lenia.params['m'], this.lenia.params['s'])
            const scalarfield =tf.mul(dt,field)
            const grid_new = tf.add(this.grid,scalarfield)
            const dummy =  grid_new.clipByValue(0,1)
            this.grid.assign(dummy)
            tf.dispose([complexgrid,fftngrid,potential_FFT,complexU,U,field,scalarfield,grid_new,dummy])
            }
            repeatAlongAxis(tensor, axis, repeats) {
                return tf.tidy(() => {
                    // Input validation
                    if (!(tensor instanceof tf.Tensor)) {
                        throw new Error('repeatAlongAxis: Input must be a Tensor');
                    }
                    if (typeof axis !== 'number' || !Number.isInteger(axis)) {
                        throw new Error('repeatAlongAxis: axis must be an integer');
                    }
                    if (axis < 0 || axis >= tensor.rank) {
                        throw new Error(`repeatAlongAxis: axis out of bounds (0-${tensor.rank - 1}), got ${axis}`);
                    }
                    if (typeof repeats !== 'number' || !Number.isInteger(repeats) || repeats < 1) {
                        throw new Error('repeatAlongAxis: repeats must be a positive integer');
                    }
            
                    const expanded = tensor.expandDims(axis + 1);
                    const tiling = new Array(expanded.rank).fill(1);
                    tiling[axis + 1] = repeats;
                    const tiled = expanded.tile(tiling);
                    const newShape = [...tensor.shape];  // Create a copy
                    newShape[axis] *= repeats;
                    
                    return tiled.reshape(newShape);
                });
            }
            
            zoom(tensor, zoom) {
                return tf.tidy(() => {
                    // Input validation
                    if (!(tensor instanceof tf.Tensor)) {
                        throw new Error('zoom: Input must be a Tensor');
                    }
                    if (typeof zoom !== 'number' || !Number.isInteger(zoom) || zoom < 1) {
                        throw new Error('zoom: zoom factor must be a positive integer');
                    }
            
                    let currentTensor = tensor;
                    for (let i = 0; i < tensor.rank; i++) {
                        currentTensor = this.repeatAlongAxis(currentTensor, i, zoom);
                    }
                    return currentTensor;
                });
            }
        Condense(){
            return tf.tidy(()=>{ 
                const Zaxis = this.grid.shape[0]
                const CondensedGrid = this.grid.sum(0)
                const NormalisedGrid = CondensedGrid.div(Zaxis)
                return NormalisedGrid})
            }
        updateGridSize() {
            tf.dispose([this.gridVoxel, this.ekernel, this.kernel]);
            const dummy = this.resize(this.grid, this.lenia.size);
            this.grid.dispose()
            this.grid = tf.variable(dummy);
            this.init();


        }
        applyColormap(matrix) {
            return tf.tidy(() => {
            const colourbarmax = tf.tensor1d([253 / 255, 232 / 255, 64 / 255]);
            const colourbarmin = tf.tensor1d([68 / 255, 13 / 255, 84 / 255]); 
            const backgroundColor = tf.tensor1d([25,25,26]);
            const expandedMatrix = matrix.expandDims(-1);
            const mask = tf.greater(expandedMatrix ,0)
            const background = tf.mul(tf.equal(mask,0),backgroundColor)
            const diff = colourbarmax.sub(colourbarmin);
            const rgbTensor = expandedMatrix.mul(diff).add(colourbarmin);
            
        
            return tf.add(tf.mul(rgbTensor,mask),background)});
        }
        rgb2rgba(rgb){
            return tf.tidy(() => {
            const squeezed = rgb.shape.length === 4 ? tf.squeeze(rgb) : rgb; // remove batch dim if present
            const [r, g, b] = tf.split(squeezed, 3, 2); // split rgb into separate tensors
            const alpha = tf.ones([squeezed.shape[0], squeezed.shape[1], 1], 'float32'); // create alpha channel tensor once // note its only done once so tensor shape must be constant
            const rgba = tf.stack([r, g, b, alpha], 2); // restack r+g+b+alpha to rgba
            tf.dispose([squeezed, r, g, b]);
            return rgba;})
        }
        g2rgba(g){
        return tf.tidy(() => {    
        const rgb = this.applyColormap(g)
        return this.rgb2rgba(rgb)})
        }

    }

        class mesh {
            constructor(lenia) {
            this.lenia = lenia;
            this.canvas = customCanvas
            // Volume data
            this.volumeData = this.lenia.tensor.grid;
            this.volumeDims = this.volumeData.shape; // [width, height, depth]
        
            // Camera parameters
            this.cameraPos = [100, 100, 100];
            this.cameraTarget = this.volumeDims.map(x => x / 2);
            this.fov = 45;   
            const halfFovTan = Math.tan((this.fov* Math.PI) / 180 / 2);
            this.sxy = [halfFovTan, halfFovTan]; 
            this.width  = this.canvas.width;
            this.height = this.canvas.height;
            // Orbital control parameters
            this.isDragging = false;
            this.lastMouseX = 0;
            this.lastMouseY = 0;
            this.radius = 0;      // Distance from camera to target
            this.theta = 0;       // Azimuthal angle
            this.phi = Math.PI/2; // Polar angle
    
            // Initialize camera spherical coordinates
            this.computeSpherical();
    
            this.registerVolumeRayMarchKernel();
            this.updateVectors();
    
            // Event listeners for orbital controls
            this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
            this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
            this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
            this.canvas.addEventListener('wheel', this.onMouseWheel.bind(this));
            this.canvas.addEventListener('resize', () => this.render());
        }
    
        // ... (keep existing methods unchanged until updateVectors)
    
        // [Add new methods for orbital controls below]
    
        computeSpherical() {
            // Convert camera position to spherical coordinates relative to target
            const dx = this.cameraPos[0] - this.cameraTarget[0];
            const dy = this.cameraPos[1] - this.cameraTarget[1];
            const dz = this.cameraPos[2] - this.cameraTarget[2];
            
            this.radius = Math.sqrt(dx*dx + dy*dy + dz*dz);
            this.theta = Math.atan2(dz, dx);       // Azimuthal angle around Y axis
            this.phi = Math.acos(dy / this.radius); // Polar angle
        }
    
        updateCameraPosition() {
            // Convert spherical coordinates to Cartesian
            const x = this.radius * Math.sin(this.phi) * Math.cos(this.theta);
            const y = this.radius * Math.cos(this.phi); // Correct Y
            const z = this.radius * Math.sin(this.phi) * Math.sin(this.theta); // Correct Z
            this.cameraPos = [
            this.cameraTarget[1] + x,
            this.cameraTarget[0] + y,
            this.cameraTarget[2] + z
            ];
        }
    
        onMouseDown(event) {
            this.isDragging = true;
            this.lastMouseX = event.clientX;
            this.lastMouseY = event.clientY;
            event.preventDefault();
        }
    
        onMouseUp(event) {
            this.isDragging = false;
            event.preventDefault();
        }
    
        onMouseMove(event) {
            if (!this.isDragging) return;
            
            const deltaX = event.clientX - this.lastMouseX;
            const deltaY = event.clientY - this.lastMouseY;
            this.lastMouseX = event.clientX;
            this.lastMouseY = event.clientY;
    
            if (event.shiftKey) {
                this.pan(deltaX, deltaY);
            } else {
                this.rotate(deltaX, deltaY);
            }
    
            this.updateVectors();
            this.render();
            event.preventDefault();
        }
    
        onMouseWheel(event) {
            this.zoom(event.deltaY);
            this.updateVectors();
            this.render();
            event.preventDefault();
        }
    
        rotate(deltaX, deltaY) {
            const sensitivity = 0.01;
            this.theta += deltaY * sensitivity;
            this.phi -= deltaX * sensitivity;
    
            // Keep phi between 0.1π and 0.9π to prevent flipping
            this.phi = Math.max(0.1 * Math.PI, Math.min(0.9 * Math.PI, this.phi));
            
            this.updateCameraPosition();
        }
    
        pan(deltaX, deltaY) {
            const sensitivity = 0.005 * this.radius;
            // FIX: Remove negative signs from deltaX and add to deltaY
            const panX = this.right[0] * deltaX * sensitivity + this.up[0] * deltaY * sensitivity;
            const panY = this.right[1] * deltaX * sensitivity + this.up[1] * (-deltaY) * sensitivity;
            const panZ = this.right[2] * deltaX * sensitivity + this.up[2] * (-deltaY) * sensitivity;
        
            this.cameraTarget[1] += panX;
            this.cameraTarget[0] += panY;
            this.cameraTarget[2] += panZ;
            this.cameraPos[1] += panX;
            this.cameraPos[0] += panY;
            this.cameraPos[2] += panZ;
        }
    
        zoom(deltaY) {
            const sensitivity = 0.001;
            this.radius *= 1 + deltaY * sensitivity;
            this.radius = Math.max(0.1, Math.min(this.radius, 1000));
            this.updateCameraPosition();
        }
        
            registerVolumeRayMarchKernel() {
            tf.registerKernel({
                kernelName: 'SolidColorKernel',
                backendName: 'webgl',
                kernelFunc: ({ inputs, backend,attrs }) => {
                // Now, attrs is a nested array in the order of your customUniforms:
                // [ volumeDims, cameraPos, resolution, sxy, f, r, u ]
                const [volumeDims, cameraPos, resolution, sxy, f, r, u] = attrs;
                const inputTensors = [inputs.volumeData]; // Match input key
        
                const program = {
                    variableNames: ['volumeData'], // Match input key
                    customUniforms: [
                    { type: 'vec3', name: 'volumeDims' },
                    { type: 'vec3', name: 'cameraPos' },
                    { type: 'vec2', name: 'resolution' },
                    { type: 'vec2', name: 'sxy' },
                    { type: 'vec3', name: 'f' },
                    { type: 'vec3', name: 'r' },
                    { type: 'vec3', name: 'u' }
                    ],
                    // Use the uniform for outputShape:
                    outputShape: resolution,
                    userCode: `
                    // Custom 3D volume ray marching implementation
        
                    vec2 intersectBox(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax) {
                        vec3 tMin = (boxMin - ro) / rd;
                        vec3 tMax = (boxMax - ro) / rd;
                        vec3 t1 = min(tMin, tMax);
                        vec3 t2 = max(tMin, tMax);
                        float tNear = max(max(t1.x, t1.y), t1.z);
                        float tFar = min(min(t2.x, t2.y), t2.z);
                        return vec2(tNear, tFar);
                    }
                     float getChannel(vec4 frag, ivec2 innerCoord) {
                        // Same as before - correct channel mapping
                        if (innerCoord.x == 0) {
                            return (innerCoord.y == 0) ? frag.r : frag.g;
                        } else {
                            return (innerCoord.y == 0) ? frag.g : frag.g;
                        }
                    }

                float sampleVolume(vec3 pos) {
                    ivec3 voxelCoord = ivec3(pos);
                    int X = int(volumeDims.x);
                    int Y = int(volumeDims.y);
                    int Z = int(volumeDims.z);

                    // Calculate UVs with proper scaling for odd dimensions
                    float u = (float(voxelCoord.x) + 0.5) / float(X); // Directly map X to [0,1]
                    float v = (float(voxelCoord.z * Y + voxelCoord.y) + 0.5) / float(Y * Z); // Flatten Y and Z

                    return texture(volumeData, vec2(u, v)).r;
                }
               vec3 computeRayDirection(vec2 coords, vec2 resolution, vec2 sxy, vec3 f, vec3 r, vec3 u) {
                    // Center pixel coordinates by adding 0.5 before normalization
                    vec2 normcoords = (coords + 0.5) / resolution;
                    
                    // Create aspect ratio-corrected normalized device coordinates
                    float aspect = resolution.x / resolution.y;
                    vec2 ndc = vec2(
                        (2.0 * normcoords.x - 1.0) * aspect,  // Apply aspect ratio to X coordinate
                        (1.0 - 2.0 * normcoords.y)
                    );
                    
                    // Scale by FOV factors and combine basis vectors
                    vec2 sndc = ndc * sxy;
                    return normalize(sndc.x * r + sndc.y * u + f);
                }
                        
                    void main() {
                        ivec2 outputCoords = getOutputCoords();

                        vec3 ro = cameraPos;
                        vec3 rd = computeRayDirection(vec2(outputCoords), resolution, sxy, f, r, u); 

                        vec2 t = intersectBox(ro, rd, vec3(0.0), volumeDims);
                        if (t.y < 0.0 || t.x > t.y) { 
                            setOutput(0.0);
                            return;
                        }

                        // Ensure we start inside the box.
                        t.x = max(t.x, 0.0);
                        vec3 v = ro + rd * t.x;
                        vec3 rdSign = sign(rd);
                        vec3 tDelta = 1.0 / abs(rd);
                        vec3 s = (rdSign * (floor(v) - v + 0.5) + 0.5) * tDelta;

                        float tcurr = 0.0;
                        while (tcurr < (t.y - t.x)) {
                            // Determine the next voxel boundary to hit
                            vec3 mask = vec3(lessThanEqual(s, min(s.yzx, s.zxy)));
                            float nextStep = min(min(s.x, s.y), s.z);
                        
                            vec3 voxelPos =  floor(v + 1e-6) + 0.5;;
                            
                            // Sample the volume at the current voxel position.
                            float samplev = sampleVolume(voxelPos);
                            if (samplev != 0.0) {
                                outputColor = vec4(vec3(samplev), 1.0);
                                return;
                            }
                            
                            // Step to the next voxel boundary.
                            tcurr = nextStep;
                            s += tDelta * mask;
                            v += rdSign * mask;
                        }

                        // No non-zero sample was found; output black.
                        outputColor = vec4(0.0, 0.0, 0.0, 1.0);
                        }
                    `,
                };
        
                // Pass the custom uniforms (which are already a nested array)
                return backend.compileAndRun(
                    program,
                    inputTensors,
                    'float32',
                    attrs
                );
                },
            });
            }
        
            onResize() {
                this.canvas.width = Math.floor(this.canvas.clientWidth);
                this.canvas.height = Math.floor(this.canvas.clientHeight);
                if (this.canvas.width === 0 && this.canvas.height === 0) return;
                
                if (this.canvas.width !== this.width || this.canvas.height !== this.height) {
                    this.width = this.canvas.width;
                    this.height = this.canvas.height;
                    
                }
            }
            updateVectors() {
        // Compute forward = (target - position).normalized
        this.forward = this.normalize([
            this.cameraTarget[0] - this.cameraPos[0],
            this.cameraTarget[1] - this.cameraPos[1],
            this.cameraTarget[2] - this.cameraPos[2]
        ]);

        // Correct right vector: worldUp × forward
        const worldUp = [0,1,0];
        this.right = this.normalize(this.cross(worldUp, this.forward));

        // CORRECTED: Compute up as forward × right (not right × forward)
        this.up = this.normalize(this.cross(this.forward, this.right)); // Fix order
    }
        
            // calculate the determinant of a matrix m
            det(m) {
                return tf.tidy(() => {
                    const [r, _] = m.shape;
                    if (r === 2) {
                        const t = m.as1D();
                        const a = t.slice([0], [1]).dataSync()[0];
                        const b = t.slice([1], [1]).dataSync()[0];
                        const c = t.slice([2], [1]).dataSync()[0];
                        const d = t.slice([3], [1]).dataSync()[0];
                        const result = a * d - b * c;
                        return result;
                    } else {
                        let s = 0;
                        const rows = [...Array(r).keys()];
                        for (let i = 0; i < r; i++) {
                            const rowIndices = rows.filter(e => e !== i);
                            const sub_m = m.gather(tf.tensor1d(rowIndices, 'int32'));
                            const sli = sub_m.slice([0, 1], [r - 1, r - 1]);
                            const element = m.slice([i, 0], [1, 1]).dataSync()[0];
                            s += Math.pow(-1, i) * element * this.det(sli);
                        }
                        return s;
                    }
                });
            }
            
            invertMatrix(m) {
                return tf.tidy(() => {
                    const d = this.det(m);
                    if (d === 0) {
                        console.log("Matrix is singular, cannot invert.");
                        return null;
                    }
                    const [r, _] = m.shape;
                    const rows = [...Array(r).keys()];
                    const dets = [];
                    for (let i = 0; i < r; i++) {
                        for (let j = 0; j < r; j++) {
                            const rowIndices = rows.filter(e => e !== i);
                            const sub_m = m.gather(tf.tensor1d(rowIndices, 'int32'));
                            let sli;
                            if (j === 0) {
                                sli = sub_m.slice([0, 1], [r - 1, r - 1]);
                            } else if (j === r - 1) {
                                sli = sub_m.slice([0, 0], [r - 1, r - 1]);
                            } else {
                                const [a, b, c] = tf.split(sub_m, [j, 1, r - (j + 1)], 1);
                                sli = tf.concat([a, c], 1);
                            }
                            const minorDet = this.det(sli);
                            dets.push(Math.pow(-1, i + j) * minorDet);
                        }
                    }
                    const com = tf.tensor2d(dets, [r, r]);
                    const tr_com = com.transpose();
                    const inv_m = tr_com.div(tf.scalar(d));
                    return inv_m;
                });
            }   
            volumeRayMarchOp(volumeData, volumeDims, cameraPos, resolution, sxy, f, r, u) {
                return tf.engine().runKernel(
                'SolidColorKernel',
                { volumeData },
                [
                    volumeDims, // vec3
                    cameraPos,  // vec3
                    resolution, // vec2
                    sxy,        // vec2
                    f,          // vec3
                    r,          // vec3
                    u           // vec3
                ]
                );
            }
            async render() {
            this.onResize()
            // Wrap the computations in a tf.tidy so that every temporary tensor is automatically disposed.
            let processer;
            if (this.lenia.render === "3D") {
                const output = tf.tidy(() => {
                    return this.volumeRayMarchOp(
                        this.volumeData,
                        this.volumeDims,
                        this.cameraPos,
                        [this.height, this.width],
                        this.sxy,
                        this.forward,
                        this.right,
                        this.up
                    );
                });
                const rgbaTensor = this.lenia.tensor.g2rgba(output);
                const gpuData = rgbaTensor.dataToGPU({ customTexShape: [this.height,this.width] });
                processer = drawTexture(customCanvas, gpuData.texture, { format: 'rgba' });;

                // Wait for WebGL to sync before finishing render.
                await syncWait(processer.gl);
                tf.dispose([output,rgbaTensor,gpuData])
            } else if (this.lenia.render === "2D") {
                const output = this.lenia.tensor.Condense();
                const doutput = this.Dimage(output)

                const rgbaTensor = this.lenia.tensor.g2rgba(doutput);

                const gpuData = rgbaTensor.dataToGPU({ customTexShape: [this.height,this.width] });

                processer = drawTexture(customCanvas, gpuData.texture, { format: 'rgba' });;

                // Wait for WebGL to sync before finishing render.
                await syncWait(processer.gl);
                tf.dispose([output,zoomed,rgbaTensor,gpuData])
    
            }
            }
            Dimage(output){
                const [_,leniaHeight,leniaWidth] = this.lenia.size; 
                const resolution = [this.canvas.clientHeight,this.canvas.clientWidth]
                console.log('resolution',resolution[1]/resolution[0])
                const leniaresolution = [leniaHeight,leniaWidth]

                const scaling = resolution.map((x,i)=>x/leniaresolution[i])
                const sortedScalingIndices = scaling.map((_, i) => i).sort((a, b) => scaling[a] - scaling[b]);
                const [smallerScaleIndex, largerScaleIndex] = sortedScalingIndices;
                const additionalScale = resolution[largerScaleIndex]/resolution[smallerScaleIndex];
                const scaledDim = leniaresolution[smallerScaleIndex]*additionalScale
                const Dimension = ['height','width']
                this.canvas[Dimension[smallerScaleIndex]] = this.lenia.size[smallerScaleIndex]
                this.canvas[Dimension[largerScaleIndex]] = Math.floor(scaledDim)
                this.height = this.canvas.height
                this.width = this.canvas.width

                const tilingFactors = [1, 1];
                tilingFactors[largerScaleIndex] = Math.ceil(additionalScale);
                console.log('tilingFactor',tilingFactors)
                const tiledOutput = output.tile(tilingFactors);
                const shift = [0, 0].map((x, i) => {
                    if (i === largerScaleIndex) {
                      return Math.floor(scaledDim / 2);
                    }
                    return x;
                  });
                const rolledOutput = this.lenia.tensor.roll(tiledOutput,[0,1],shift)
                return tf.slice2d(rolledOutput,[0,0],[this.lenia.size[smallerScaleIndex],Math.floor(scaledDim)])

            }

            normalize(v) {
            const len = Math.hypot(...v);
            return [v[0] / len, v[1] / len, v[2] / len];
            }

            cross(a, b) {
            return [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
            ];
            }

            dot(a, b) {
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            }
            async init() {
            if (!this.initialized) {
                await this.initializationPromise;
                this.canvas.style.display = 'block'
            }
            this.update();
            }

            async update() {
            if (!this.initialized) {
                await this.initializationPromise;
            }
            // Proceed with rendering logic
            await this.render();
            }


        }
    class UI{
        constructor(lenia){
        this.lenia = lenia

        //loading screen

        this.loading = document.getElementById('loading')

        // Core Elements
        this.main = document.getElementById('main');
        this.menu = document.getElementById('menu');
        this.menutopbar = document.getElementById('menu-topbar')
        this.sidebar = document.querySelector('.sidebar');
        this.sidebarmenu = document.querySelector('.sidebarmenu');
        this.playcontrol = document.getElementById('playcontrol');

        this.sidebarState = '';
        this.layoutState = '';

        this.aspectRatio = window.innerWidth / window.innerHeight;
        this.sidebarState = 'reduced'
        this.layoutState = this.aspectRatio > 0.75 ? 'row' : 'column';
        this.toggleLayoutbutton = document.getElementById('toggleLayout')
        this.updateLayoutDimensions();



        // Display Elements
        this.gencounter = document.getElementById("gen");
        this.timecounter = document.getElementById("time");
        this.framecounter = document.getElementById("frame")
        
        this.seedinput = document.getElementById('seedinput')
        this.seeddisplay = document.getElementById("seed");


        // pregenerated menu items
        this.menuItems = document.querySelectorAll('.menu li a');

        // menu containers
        this.containers = document.querySelectorAll('.container');
        this.iseditmode = true;
        this.menumode = 'pregenerated';
        this.menumodes = {'pregenerated':[['Pregenerated Patterns',true],['AnimalWindow']],
                            'generate':[['Create New Lenia',true],['seed-input-container',
                                        'dimensions-container',
                                        'generate-container',
                                        'parameters-container',
                                        'generate-button']],
                            'edit':[ ['Edit',false]   ,['dimensions-container',
                                        'draw-container',
                                        'parameters-container',
                                        'generate-button']]

                            
        }




        this.playButtons = document.querySelectorAll('#play-button-checkbox');
        this.isbuttonchecked = false;
        this.playButtons.forEach((button) => {
            button.checked = false; 
            button.addEventListener('change', () => {
                const allChecked = button.checked;
        
                this.playButtons.forEach((btn) => {
                    btn.checked = allChecked; 
                });

                if (allChecked && !this.isbuttonchecked) {
                    this.isbuttonchecked = true;
                    this.lenia.startstate();
                } else if (!allChecked && this.isbuttonchecked) {
                    this.isbuttonchecked = false;
                    this.lenia.stopstate()
                }
            });
        });


        //Grid Size Parameters
            this.dimcounter = document.getElementById("dim");
            this.gridparamscontainer = document.getElementById("dimensions-container");
        // Random Grid 
            

            //Density
                this.BindAndSyncInputs("density",this.lenia.cluster_density,[0,1,0.001])

        // Kernel Parameters
            this.radiuscounter = document.getElementById("radius");

            //mu
                this.BindAndSyncInputs("mu",this.lenia.params['m'],[0,1,0.001])
            //sigma
                this.BindAndSyncInputs("sigma",this.lenia.params['s'],[0,1,0.001])
            //b
                this.bcontainer = document.getElementById("bcontainer");
                this.bdimcounter = document.getElementById("bdim");
                this.bdimaddbutton = document.getElementById("bdim-add-button");
                this.bdimsubbutton = document.getElementById("bdim-sub-button");           


        this.PopulateAnimalList()
        this.clustermin = document.getElementById("cluster-min-value");
        this.clustermax = document.getElementById("cluster-max-value");
        this.clusterslidermin = document.getElementById("min");
        this.clusterslidermax = document.getElementById("max");
        this.bindDualInputs([this.clustermin,this.clustermax],[this.clusterslidermin,this.clusterslidermax],lenia.cluster_range,lenia.cluster_range)

        this.tbuttons =  Array.from(document.querySelectorAll('.time-bar .button'));
        this.timeres = document.getElementById('timeres');

    }   
    editMenu(children) {
        if (!this.menu || !this.menu.children) {
            console.error('Invalid menu');
            return;
        }
        const headerArray = Array.from(this.menutopbar.children);
        const childrenArray = Array.from(this.menu.children);

        childrenArray.forEach(child => {
            child.style.display = 'none';
        })
        headerArray[0].innerText = children[0][0]
        headerArray[1].style.display = children[0][1] ? 'flex' : 'none'

        children[1].forEach(childId => {
            const matchingChild = this.menu.querySelector(`#${childId}`);
            if (matchingChild) {
                matchingChild.style.display = 'block';
            } else {
                console.warn(`Child with id "${childId}" not found in container.`);
            }
        });
    }
        toggleMenu(layout){
            this.menumode = layout
            const layoutList = this.menumodes[this.menumode]
            this.iseditmode = layout === 'generate'
            this.updatedim()
            this.editMenu(layoutList) 
        }

        updatedensity(){
            this.densityslider.value = this.lenia.cluster_density
            this.densitytext.value = this.lenia.cluster_density
        }

            

        toggleSidebar() {
            const transitions = {
                'open': { nextState: 'reduced', width: '140px' },
                'reduced': { nextState: 'collapsed', width: '41px' },
                'collapsed': { nextState: 'open', width: '0px' }
            };
            const nextInfo = transitions[this.sidebarState];
            this.sidebarmenu.style.width = nextInfo.width;
            this.sidebarState = nextInfo.nextState;
        }

        toggleLayout() {
            this.layoutState = (this.layoutState === 'row') ? 'column' : 'row';
            this.updateLayoutDimensions();
            this.lenia.canvasResize();
        }

        rowLayout() {
        if (this.main.classList.contains('row-main')) return;
        else if (this.main.classList.contains('col-main')) {
            this.main.classList.replace('col-main', 'row-main');}
        else {
            this.main.classList.add('row-main');}
        }
        columnLayout() {
        if (this.main.classList.contains('col-main')) return;
        else if (this.main.classList.contains('row-main')) {
            this.main.classList.replace('row-main', 'col-main');}
        else {
            this.main.classList.add('col-main');}
        }

        updateLayoutDimensions() {
            const icons = Array.from( this.toggleLayoutbutton.children)
            icons[0].style.display = (this.layoutState === 'row') ? 'none' : 'block'
            icons[1].style.display = (this.layoutState === 'row') ? 'block' : 'none'
            if (this.layoutState === 'row') {
                this.sidebar.style.position = 'relative'
                this.rowLayout();
            } else {
                this.sidebar.style.position = 'fixed'
                this.columnLayout();
            }

        }

        toggleRenderDimension(button) {
            const D = button.innerText
            button.style.backgroundColor = "hsl(var(--primary))";
            button.style.color = "hsl(var(--foreground))"
            const otherD = (button.innerText== '2D') ? '3D' : '2D'
            const otherID = "render-dimension-button-" + otherD
            const otherButton = document.getElementById(otherID)
            otherButton.style.backgroundColor = "hsl(var(--muted))"
            otherButton.style.color = "hsl(var(--muted-foreground))"
            this.lenia.render = button.innerText
            this.lenia.mesh.render();
        }

        validateData(value, type, slice = 0) {
            // Convert value to string if it's not already
            value = String(value);
        
            // Determine allowed characters and handle dots based on type
            let allowedChars;
            let dotAllowed = false;
            if (type === 'int') {
                allowedChars = '0-9';
            } else if (type === 'float') {
                allowedChars = '0-9.';
                dotAllowed = true;
            } else {
                throw new Error('Invalid type specified');
            }
            // Remove any characters not allowed
            value = value.replace(new RegExp(`[^${allowedChars}]`, 'g'), '');
        
            if (type === 'float' && dotAllowed) {
                // Ensure only one dot is present
                const dotIndex = value.indexOf('.');
                if (dotIndex !== -1) {
                    const integerPart = value.slice(0, dotIndex);
                    const decimalPart = value.slice(dotIndex + 1);
                    value = `${integerPart}.${decimalPart.replace(/\./g, '')}`;
                }
            }
            if (type === 'float' && value.includes('.') && slice > 0) {
                // Truncate the decimal part to fit the slice
                let [integerPart, decimalPart] = value.split('.');
                decimalPart = decimalPart.slice(0, slice - integerPart.length);
                value = `${integerPart}.${decimalPart}`;
            } else if (slice > 0) {
                // Slice the value to the desired length
                value = value.slice(0, slice);
            }
            return value;
        }
        validateInput(input, type, numberRange) {
            let value = input.value;
            if (type === 'int') {
                // Remove non-numeric characters and limit to 3 digits
                const maxDigits = numberRange[1].toString().length;
                value = this.validateData(value, 'int', maxDigits);
                value = parseInt(value, 10);
            } else if (type === 'float') {
                // Remove non-numeric characters except for '.'
                value = this.validateData(value, 'float', 5);
                value = parseFloat(value);
            }
            if (!isNaN(value)) {
                // Ensure the value is within the specified range
                value = Math.min(Math.max(value, numberRange[0]), numberRange[1]);
                input.value = value;
            } else {
                // Clear the input if the value is not a valid number
                input.value = '';
            }
        }
        validateCluster(texts,index, ranges) {
            const type = (Number.isInteger(ranges[2])) ? 'int' : 'float';
            const otherIndex = !index ? 1 : 0;
            const adjustedRange = index  ? [Math.max(texts[otherIndex].value, ranges[0]), ranges[1]] : [ranges[0], Math.min(texts[otherIndex].value, ranges[1])]
            this.validateInput(texts[index], type, adjustedRange)
        }



        menuvar() {
            this.menuItems.forEach((item, index) => {
                item.addEventListener('click', (e) => {
                    e.preventDefault();
                    var container = this.containers[index];
        
                    if (container.style.display === 'block') {
                        container.style.display = 'none';
                    } else {
                        this.containers.forEach((c) => {
                            c.style.display = 'none';
                        });
                        container.style.display = 'block';
                    }
                });
            });
        }
        updateradius() {this.radiuscounter.textContent = this.lenia.params['R'].toFixed(0);}
            updatet() {
                const value = this.lenia.params['T'];
                for (let i = 0; i < this.tbuttons.length; i++) {
                    if (value === parseInt(this.tbuttons[i].innerText)) {
                        this.tbuttons[i].classList.add('active');
                    } else {
                        this.tbuttons[i].classList.remove('active');
                    }
                    this.timeres.value = value;
                }
            }
        updatedim(){
                this.dimcounter.textContent = this.lenia.DIM;
                this.gridparamscontainer.textContent = '';

                let values = this.lenia.size
                let eventHandlers = [this.lenia.updateGridSize,this.lenia.renderBounds,this.lenia.renderRemoveBounds,this.lenia.renderRemoveBounds]
                let eventTypes = ["change","input","mouseup","mouseend"]


                if (this.iseditmode) {
                    values = this.lenia.size.map((_, i) => [this.lenia.cluster_size[i], this.lenia.size[i]]);
                    eventHandlers = [Array(eventHandlers.length).fill(null),eventHandlers]
                    eventTypes =[Array(eventTypes.length).fill(null),eventTypes]
                }

                this.populateParameters(this.gridparamscontainer,["X","Y","Z"],values,[0,128,1],"parameter-row dimension-row",eventHandlers,eventTypes)

        }   
        updatem() { const value = this.lenia.params['m'];
                    this.mutext.value = value;
                    this.muslider.value  = value;}
        updates() {const value = this.lenia.params['s'];
                    this.sigmatext.value = value;
                    this.sigmaslider.value  = value;}

        updateb() {const values = this.lenia.params['b'];
                    this.bdimcounter.textContent = values.length;
                    this.bcontainer.innerHTML = '';
                    const range  = Array.from({ length: values.length + 1 }, (_, index) => index);
                    this.populateParameters(this.bcontainer,range,values,[0,1,0.001],"parameter-row beta-row",this.lenia.tensor.generateKernel.bind(this.lenia.tensor),"change");
                    this.lenia.tensor.generateKernel();}
        updateparams() {
            this.updateradius();
            this.updatet();
            this.updatem();
            this.updates();
            this.updateb();
            this.updatedim();
        }
        updatetime(){this.timecounter.textContent = this.lenia.time.toFixed(2)+'s';}
        updateframe(){this.framecounter.textContent = 'f ' + this.lenia.framecounter;}
        updategen(){this.gencounter.textContent = 'g ' + this.lenia.gen;}

        updateseed() {
            this.seeddisplay.textContent = this.lenia.seed ? 'seed:' + this.lenia.seed : 'id: ' + this.lenia.id 
        }
        updatename() {
            if (window.getComputedStyle(this.seedcontainer).display === "block") {
                this.seedcontainer.style.display = "none";
            }
            if (window.getComputedStyle(this.namecontainer).display === "none") {
                this.namecontainer.style.display = "flex";
            }
            this.typedisplay.textContent = this.type;
            this.namedisplay.textContent = this.name;
        }
        updateColorbar = () => {
            this.lenia.colourbarmin = this.HextoRGB(this.colorbarmin.value)
            this.lenia.colourbarmax = this.HextoRGB(this.colorbarmax.value)
        }
        PopulateAnimalList() {
            if (!animalArr) return;
            var list = document.getElementById("AnimalList");
            if (!list) return;
            list.innerHTML = "";
            var lastCode = "";
            var lastEng0 = "";
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
                        li.classList.add("parameter-row");
                        const text = li.appendChild(document.createElement("DIV"));
                        text.style.position = "absolute";
                        text.style.left = "0";
                        text.style.marginLeft = "2vw";
                        const code = li.appendChild(document.createElement("DIV"));
                        code.style.position = "absolute";
                        code.style.right = "0";
                        text.title = a[0] + " " + engSt.join(" ") + "\n" + ruleSt;
                        if (sameCode) codeSt = " ".repeat(codeSt.length);
                        if (sameEng0) engSt[0] = lastEng0.substring(0, 1) + ".";
                        text.innerHTML = engSt.join(" ");
                        li.style.color = "#cdd0d6"
                        li.style.width = "calc(100% - 2vw)";
                        code.innerHTML = codeSt;
                        li.dataset["animalid"] = i;
                        li.addEventListener("click", this.SelectAnimalItem);
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
                        div.classList.add("parameter-row");
                        const text = div.appendChild(document.createElement("DIV"));
                        text.style.position = "absolute";
                        text.style.left = "0";
                        text.style.marginLeft = "1vw";
                        const arrow = div.appendChild(document.createElement("DIV"));
                        arrow.classList.add("arrow"); 
                        text.title = engSt;
                        text.innerHTML = engSt[engSt.length-1];
                        const scalar = Math.pow(8/9, nextLevel)
                        const fontsize = scalar*20
                        const padding = scalar*3
                        div.style.fontSize = `${fontsize}px`
                        div.style.color = "#cdd0d6"
                        text.style.paddingBottom = `${padding}%`
                        div.style.paddingTop = `${padding}%`
                        div.addEventListener("click", function (e) {
                            this.parentElement.classList.toggle("closed");
                            arrow.classList.toggle("sideways"); // Add this line to toggle the sideways class on the arrow
                        });
                        for (var k=0; k<foreNum; k++) {
                            node = node.appendChild(document.createElement("UL"));
                        }
                        currLevel = nextLevel;
                    }
                }
            }
        }
        GetLayerSt(st) {
            if (!st) return "";
            var s = st.split(/\=|\;|\*/);
            var s3 = s[3].split(/\(|\)|\,/);
            var l = Math.max(s3.length - 3, 0);
            return "[" + (l+1) + "]";
        }
        SelectAnimalItem = (e) => {
        const rundummy = this.lenia.isRunning
        if (rundummy) {this.lenia.updatestate()}
            var item = e.target;
            if (!item.dataset["animalid"]) item = item.parentElement;
            var id = parseInt(item.dataset["animalid"]);
            this.lenia.tensor.load.SelectAnimalID(id);
        if (rundummy) {this.lenia.updatestate()}
        }
            
        populateParameters(container, labels, values, ranges, classnames, eventhandlers,eventTypes) {
            values.forEach((_, i) => {
                this.addParameter(container, labels, values, i, ranges, classnames, eventhandlers,eventTypes)}
            );
        }
        
        
        addParameter(container, names, values, index, ranges, classnames, eventhandlers,eventTypes) {
            const sliceValue = Number.isInteger(ranges[2]) ? ranges[1].toString().length : 5;
            const type = Number.isInteger(ranges[2]) ? 'int' : 'float'
        
            const row = document.createElement("div");
            row.className = classnames;
        
            const label = document.createElement("label");
            label.className = "paras-label";
            label.textContent = names[index].toString();
            row.appendChild(label);
        
            const createSliderInput = (value) => {
                const slider = document.createElement("input");
                slider.type = "range";
                slider.min = ranges[0];
                slider.max = ranges[1];
                slider.step = ranges[2];
                slider.value = value;
                return slider;
            };
        
            const createTextInput = (value) => {
                const text = document.createElement("input");
                text.type = "text"; 
                text.value = type == 'int' ? Math.round(value) : value
                text.className = "para-input";
                text.style.width = `${sliceValue + 1}ch`;
                return text;
            };

            const BindEvent = (text, slider,eventhandler, eventType) => {
                if (Array.isArray(eventhandler)) {
                    eventhandler.forEach((func, idx) => {
                        if (func) {
                            text.addEventListener(eventType[idx].toString(), func); 
                            slider.addEventListener(eventType[idx], (event) => {
                                this.inputIndex = index;
                                this.inputValue = parseInt(event.target.value);
                                func.bind(this.lenia)()
                            }); 
                        }
                    });
                } else {
                    text.addEventListener(eventType, eventhandler);  
                    slider.addEventListener(eventType, eventhandler); 
                }
            };  
            container.appendChild(row);
            if (Array.isArray(values[index])) {
                const sliderContainer = document.createElement("div");
                const textContainer = document.createElement("div");
                sliderContainer.className = "range-container";
                textContainer.style.minWidth = `${3*(sliceValue + 1)}ch`;
                textContainer.className = "genRangeInputContainer";
                row.appendChild(sliderContainer);
                row.appendChild(textContainer);
        
                values[index].forEach((value, idx) => {
                    const slider = createSliderInput(value);
                    slider.className = idx ? 'range max' : 'range min';;
                    const text = createTextInput(value);
                    const comma = document.createElement("span");
                    comma.textContent = ',',
                    sliderContainer.appendChild(slider);
                    textContainer.appendChild(text);
                    if (!idx) {textContainer.appendChild(comma)}    ;
                    BindEvent(text, slider, eventhandlers[idx], eventTypes[idx]);
                    
                });
        
                this.bindDualInputs( textContainer.querySelectorAll('input'), sliderContainer.childNodes, ranges, values[index]);
            } else {
                const slider = createSliderInput(values[index],true);
                const text = createTextInput(values[index]);
                row.appendChild(slider);
                slider.classList.add("single-range");
                row.appendChild(text);
                BindEvent(text, slider,eventhandlers, eventTypes);
                this.BindInputs(text,slider, ranges,values[index]);
            }
        
        }
        
        Removeparameter(container,i){
            container.removeChild(container.childNodes[i])
        }


        BindAndSyncInputs(name,values,ranges){
            const slidername = name + "slider";
            const textname = name + "text";
            this[slidername] = document.getElementById(slidername);
            this[textname] = document.getElementById(textname);
            this.BindInputs(this[textname],this[slidername], ranges,values);
        }

        BindInputs(text,slider, ranges,values) {
            let type = null;
            if (ranges[2] === 1) {type = 'int'}
            else {type = 'float'}
            const parse = (type === 'int') ? parseInt : parseFloat;
            const length = (type === 'int') ? ranges[1].toString().length : 5;

            slider.addEventListener("input", (event) => {
                const sliderValue = parse(event.target.value);
                text.value = sliderValue;
            });
            slider.addEventListener("change", (event) => {
                const sliderValue = parse(event.target.value);
                values = sliderValue
            });
            text.addEventListener("input", (event) => {
                text.value = this.validateData(event.target.value, type, length)});
        
            text.addEventListener("blur", (event) => {
                this.validateInput(event.target, type, ranges);
                const textValue = parse(event.target.value);
                slider.value = textValue; 
                values = textValue;

        });}

        bindDualInputs(texts, sliders, ranges, values = null) {
            const type = Number.isInteger(ranges[2]) ? 'int' : 'float';
            const parse = type === 'int' ? parseInt : parseFloat;
            const length = type === 'int' ? ranges[2].toString().length : 5;
        
            texts.forEach((text, index) => {
                text.addEventListener('input', () => {
                    text.value = this.validateData(text.value, type, length);
                });
                text.addEventListener('blur', () => {
                    this.validateCluster(texts, index, ranges);
                    sliders[index].value = text.value;
                    if (values) values[index] = parse(text.value);
                });
            });
        
            sliders.forEach((slider, index) => {
                slider.addEventListener('input', () => {
                    texts[index].value = slider.value;
                    const otherIndex = index ? 0 : 1;
                    slider.value = Math[index ? 'max' : 'min'](slider.value, sliders[otherIndex].value);
                    texts[index].value = slider.value;
                });
                slider.addEventListener('change', () => {
                    if (values) values[index] = parse(slider.value);
                });
            });
        }
        
        


        ParameterSync(inputs) {
            //Inputs Array
            inputs.forEach((input) => {
                input.addEventListener("input", (event) => {
                    value = event.target.value;
                })
            });
        }



        HextoRGB(hex) {
            // Remove the '#' symbol if present
            if (hex.charAt(0) === '#') {
            hex = hex.substr(1);
            }
        
            // Check if the input is a valid hexadecimal color code
            const validHexPattern = /^[0-9a-fA-F]{6}$/;
            if (!validHexPattern.test(hex)) {
            throw new Error('Invalid hexadecimal color code');
            }
        
            // Extract the individual color components
            const r = parseInt(hex.substr(0, 2), 16);
            const g = parseInt(hex.substr(2, 2), 16);
            const b = parseInt(hex.substr(4, 2), 16);
        
            // Return the RGB array
            return [r, g, b].map(x => x / 255);
        }
        RGBtoHex(rgbArray) {
            const hex = rgbArray.reduce((acc, val) => {
            const component = Math.round(val * 255).toString(16).padStart(2, '0');
            return acc + component;
            }, '');
            return `#${hex}`;
        }
        
    }


    class Load{
        constructor(lenia){
            this.lenia = lenia;
            this.DIM_DELIM = {0:'', 1:'$', 2:'%', 3:'#', 4:'@A', 5:'@B', 6:'@C', 7:'@D', 8:'@E', 9:'@F'}

        }
        SelectAnimalID(id) {
            let state = false
            if (this.lenia.isRunning) {
                state = true
                this.lenia.updatestate()}
            if (id < 0 || id >= animalArr.length) return;
            var a = animalArr[id];
            if (Object.keys(a).length < 4) return;
            const cellSt = a['cells'];
            const dummy = tf.tidy(() =>{ 
                const newgrid  = this.rle2arr(cellSt);
                return this.lenia.tensor.resize(newgrid,this.lenia.size);
            });
            this.lenia.tensor.grid.assign(dummy);

            this.lenia.UI.name = a['name']
            this.lenia.UI.type = this.SelectType(id)
            this.lenia.params = Object.assign({}, a['params'])
            this.lenia.params['b'] = this.st2fracs(this.lenia.params['b'])
            dummy.dispose();
            this.lenia.UI.updateparams()
            /*
            this.lenia.UI.updatename()*/
            this.lenia.id = id
            if (state) {this.lenia.updatestate()}
        }
        SelectType(id) {
            let names = []
            let i = id
            let currcode = animalArr[id]['code']
            while (i >= 0) {
                const a = animalArr[i]
                if (Object.keys(a).length == 3 && a['code'] !== currcode) {
                    let name = a['name']
                    name = name.split(' ').slice(1).join(' ')
                    names.push(name)
                    
                    currcode = a['code']
                }
                i--
            }
            return names.reverse().join(' ')
        }
        rle2arr(st) {
            var stacks = [];
            for (var dim = 0; dim < this.lenia.DIM; dim++) {
                stacks.push([]);
            }
            var last = '';
            var count = '';
            var delims = Object.values(this.DIM_DELIM);
            st = st.replace(/!$/, '') + this.DIM_DELIM[this.lenia.DIM - 1];
            for (var i = 0; i < st.length; i++) {
                var ch = st[i];
                if (/\d/.test(ch)) {
                    count += ch;
                } else if ('pqrstuvwxy@'.includes(ch)) {
                    last = ch;
                } else {
                    if (!delims.includes(last + ch)) {
                        this._append_stack(stacks[0], this.ch2val(last + ch) / 255, count, true);
                    } else {
                        var dim = delims.indexOf(last + ch);
                        for (var d = 0; d < dim; d++) {
                            this._append_stack(stacks[d + 1], stacks[d], count, false);
                            stacks[d] = [];
                        }
                    }
                    last = '';
                    count = '';
                }
            }
            var A = stacks[this.lenia.DIM - 1];
            var max_lens = [];
            for (var dim = 0; dim < this.lenia.DIM; dim++) {
                max_lens.push(0);
            }
            this._recur_get_max_lens(0, A, max_lens);
            this._recur_cubify(0, A, max_lens);
            return tf.tensor(A);
            }
        _append_stack(list1, list2, count, is_repeat = false) {
                list1.push(list2);
                if (count !== '') {
                    var repeated = is_repeat ? list2 : [];
                    for (var i = 0; i < parseInt(count) - 1; i++) {
                        list1.push(repeated);
                    }
                }
            }
            
        ch2val(c) {
            if (c === '.' || c === 'b') return 0;
            else if (c === 'o') return 255;
            else if (c.length === 1) return c.charCodeAt(0) - 'A'.charCodeAt(0) + 1;
            else return (c.charCodeAt(0) - 'p'.charCodeAt(0)) * 24 + (c.charCodeAt(1) - 'A'.charCodeAt(0) + 25);
            }
            
        _recur_get_max_lens(dim, list1, max_lens) {
            max_lens[dim] = Math.max(max_lens[dim], list1.length);
            if (dim < this.lenia.DIM - 1) {
                for (var i = 0; i < list1.length; i++) {
                    this._recur_get_max_lens(dim + 1, list1[i], max_lens);
                }
            }
            }
            
        _recur_cubify(dim, list1, max_lens) {
            var more = max_lens[dim] - list1.length;
            if (dim < this.lenia.DIM - 1) {
                for (var i = 0; i < more; i++) {
                    list1.push([]);
                }
                for (var i = 0; i < list1.length; i++) {
                    this._recur_cubify(dim + 1, list1[i], max_lens);
                }
            } else {
                for (var i = 0; i < more; i++) {
                    list1.push(0);
                }
            }
            }
        st2fracs(string) {
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
    }

    
    export { Lenia };

