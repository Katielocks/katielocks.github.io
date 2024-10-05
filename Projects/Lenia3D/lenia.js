tf.setBackend('webgl')






class Lenia {
    constructor() {
        this.params = {'R':15, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'kn':1, 'gn':1};
        this.DIM = 3;

        this.render = '2D'

        this.size = Array.from({ length: this.DIM }).fill(64);

        this.paramsbackup = { ...this.params };
        this.sizebackup = this.size.slice();

        this.cluster_size = this.size.map(x => x);
        this.cluster_density = 1
        this.cluster_range = [0,1]

        this.colourbarmax = [253 / 255, 232 / 255, 64 / 255];
        this.colourbarmin = [68 / 255, 13 / 255, 84 / 255];


        this.seed = null
        this.id = null
        this.isRunning = false;
        this.time = 0;
        this.gen = 0;
        this.UI = new UI(this);
        this.tensor = new tensor(this);
        this.mesh = new mesh(this);
        this.array = new array(this);
        this.renderInit();
        this.tensor.generateKernel();


        this.UI.BindParameters(this.UI.bcontainer,this.params['b'],this.tensor.generateKernel.bind(this.tensor),"change")

        this.UI.updateparams()

        this.UI.editMenu(this.UI.menu,['AnimalWindow'])

        }
        animatestate() {
            if (this.isRunning) {
              // Store the current instance of "this"
              const self = this;
              // Update the grid every 1/T seconds
              setTimeout(function() {
                self.tensor.update();
                self.renderUpdate();
                self.animatestate();
                this.gen += 1
                this.time += 1/this.params['T']
                this.UI.updatetime()
                this.UI.updategen()
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
            console.log('params',this.params)
            this.resetparams();
            console.log('params',this.params)
            this.tensor.updateGridSize();
            this.UI.updateparams();
    
            if (this.id !== null) {
                this.tensor.load.SelectAnimalID(this.id)
            }
            else{
                this.tensor.randomGrid()
                this.renderUpdate()
            }
            this.time = 0;
            this.gen = 0;
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
        

        renderInit(){
            if (this.render === '2D'){ this.array.init();; }
            else if (this.render === '3D'){ this.mesh.init(); }
        }
        renderUpdate(){
            if (this.render === '2D'){ this.array.update(); }
            else if (this.render === '3D'){ this.mesh.update(); }
        }
        renderRemove(){
            if (this.render === '2D'){ this.array.remove(); }
            else if (this.render === '3D'){ this.mesh.remove(); }
        }
        renderReset(){
            this.renderRemove()
            this.renderInit()
        }
        renderBounds(inputValue, direction) {
            if (this.render === '2D' && direction !== 2){ this.array.shouldRenderBounds = true, this.array.renderBoundsPotential = inputValue, this.array.renderBoundsDirection = (this.DIM - direction - 1) % this.DIM; 
                this.array.update()
            }
            else if (this.render === '3D'){ this.mesh.renderBounds(inputValue,direction); }
        }
        renderRemoveBounds() {
            if (this.render === '2D'){ this.array.removeBounds(); }
            else if (this.render === '3D'){ this.mesh.removeBounds(); }
        }
        updateGridSize(input,direction) {
            const values = this.size.slice().reverse();
            values[direction] = input;
            this.size = values.slice().reverse();
            this.tensor.updateGridSize();
            if (this.render === '2D'){ this.array.updateGridSize(); }
            else if (this.render === '3D'){ this.mesh.updateGridSize(); }
        }
        canvasResize() {
            
            console.log('canvas resize triggered')
            if (this.render === '2D'){ this.array.update(); }
            else if (this.render === '3D'){ this.mesh.renderContainerSync(); }
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
    transposecomplex(tensor,perms){
        return tf.tidy(()=>{
        let real = tf.real(tensor).reshape(tensor.shape).transpose(perms)
        let imag = tf.imag(tensor).reshape(tensor.shape).transpose(perms)
        return tf.complex(real,imag)
        })
    }
    
    fft1(tensor,perms){
        return  tf.tidy(() => {
            let fttensor = tf.spectral.fft(this.transposecomplex(tensor,perms))
        return this.transposecomplex(fttensor,perms)})
    }
    
    fftn(tensor){
        return tf.tidy(() => {
            let perms = tf.range(0,tensor.shape.length,1).arraySync()
            let fttensor = tensor
            for (let i=0;i<perms.length;i++){
                const realign = perms.slice();
                [realign[realign.length-1],realign[i]] = [realign[i],realign[realign.length-1]];
                fttensor =  this.fft1(fttensor,realign)
            }
            return fttensor})
    }
    
    ifft1(tensor,perms){
        return  tf.tidy(() => {
        let fttensor = tf.spectral.ifft(this.transposecomplex(tensor,perms))
        return this.transposecomplex(fttensor,perms)})
    }
    
    ifftn(tensor){
        return tf.tidy(() => {
            let perms = tf.range(0,tensor.shape.length,1).arraySync()
            let fttensor = tensor
            for (let i=0;i<perms.length;i++){
                const realign = perms.slice();
                [realign[realign.length-1],realign[realign.length-1-i]] = [realign[realign.length-1-i],realign[realign.length-1]];
                fttensor =  this.ifft1(fttensor,realign)
            }
            return fttensor})
    }

    generateKernel(){
        const SIZE = this.lenia.size
        
        const error = SIZE.map(x=>x%2)
        const mid = SIZE.map(x=>Math.floor(x/2));
        let dummy = tf.tidy(() => {
            let D = tf.zeros(SIZE);
            for (let i = 0; i < SIZE.length; i++){
                let x = tf.range(-mid[i],mid[i]+error[i],1)
                const reshapearr = Array.from(SIZE, (value, index) => (index === i ? value: 1))
                const tilearr  = Array.from(SIZE, (value, index) => (index !== i ? value: 1))
                x= x.reshape(reshapearr).tile(tilearr);
                x = tf.pow(x,2)
                D = tf.add(D,x)
            }
            D = tf.div(tf.sqrt(D),this.lenia.params['R'])
            let K = this.kernel_shell(D);
            K = this.roll(K.div(K.sum()), [0,1,2], mid)
            return this.fftn(K.cast('complex64'));
        })
        this.kernel.assign(dummy)
        dummy.dispose()
    }

    kernel_shell(r){
        return tf.tidy(() => {
        const B = this.lenia.params['b'].length
        const Br = tf.mul(B,r);
        const bs  = this.lenia.params['b']
        const b = this.map(bs,tf.minimum(tf.floor(Br).cast('int32'),B-1))
    
        const kfunc = this.kernel_core[this.lenia.params['kn'] - 1]
        return tf.mul(tf.mul(tf.less(r,1),kfunc(tf.minimum(tf.mod(Br,1),1))),b)})
    }
    map(arr,tensor){
        const tensorarr = tensor.dataSync()
        const mappedarr  = tensorarr.map(x=>arr[x])
        tensor.dispose()
        return tf.tensor(mappedarr, tensor.shape);
        }
    roll(grid, axis, shift) {
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
        this.lenia.renderUpdate()
        new_grid.dispose()
    }
    generateRandomGrid(){
        this.lenia.id = null;
        this.lenia.time = 0;
        this.lenia.gen = 0;
        this.randomGrid();
        this.lenia.renderUpdate();
        this.lenia.UI.updateseed()
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
    zoom(Tensor,zoom){
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
}

class mesh {
    constructor(lenia) {
        this.lenia = lenia;
        this.cubeProxy = new THREE.Object3D();
        this.canvas = document.getElementById('canvas-3d');
        this.generateBounds()
    }


    addEventListeners() {
        window.addEventListener('mousedown', (event) => this.onMouseDown(event));
        window.addEventListener('mousemove', (event) => this.onMouseMove(event));
        window.addEventListener('mouseup', () => this.isDrawing = false);
        window.addEventListener('resize', () => this.renderContainerSync());
    }

    onMouseDown(event) {
        this.isDrawing = true;
        this.updateMousePosition(event);
    }

    onMouseMove(event) {
        if (this.isDrawing) {
            this.updateMousePosition(event);
        }
    }
    updateMousePosition(event) {
        this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    }
    
    animate() {
            this.controls.update();
            this.renderer.render(this.scene, this.camera);
            this.needsRender = false;
        requestAnimationFrame(this.animate.bind(this));
    }
  
  
    onChange() {
      this.needsRender = true;
    }

    renderContainerSync() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;

        this.renderer.setSize(width, height,false); // Update the renderer size
    
        // Update the camera aspect ratio to match the canvas size
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
    
        this.needsRender = true; // Trigger a re-render
    }
    init() {
        this.canvas.style.display = 'block';
        this.size = this.lenia.size.slice(-3);
        this.offset = this.size.map(x => (x - 1) / 2);
        
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera( 75, this.canvas.clientWidth / this.canvas.clientHeight, 0.1, 1000 );
        this.renderer = new THREE.WebGLRenderer({ alpha: true, canvas: this.canvas });
        this.renderer.setClearColor(0x000000, 0); // 0 is the alpha value for full transparency
        
        this.camera.position.x = 0.8*64; ;
        this.camera.position.y = 0.8*64; ;
        this.camera.position.z = 0.8*64; ;
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true; 
        this.controls.dampingFactor = 0.05; // Set the damping factor for the damping effect
        
        this.addEventListeners();
        this.generateMesh();
        this.scene.add(this.mesh);
        this.renderContainerSync();
        this.update(true);
        this.animate();
    }

    generateMesh() {
        this.mesh = new THREE.InstancedMesh(
            new THREE.BoxGeometry(1, 1, 1),
            new THREE.MeshBasicMaterial(),
            this.size.reduce((a, v) => a * v)
        );

        const totalInstances = this.size.reduce((a, v) => a * v);
        const matrices = new Float32Array(totalInstances * 16); // Store matrices in a Float32Array for batching

        let index = 0;
        for (let i = 0; i < this.size[0]; i++) {
            for (let j = 0; j < this.size[1]; j++) {
                for (let k = 0; k < this.size[2]; k++) {
                    const coord = [i, j, k].map((x, idx) => x - this.offset[idx]);
                    this.cubeProxy.position.set(coord[0], coord[1], coord[2]);
                    this.cubeProxy.scale.setScalar(0); // Initially scale to 0
                    this.cubeProxy.updateMatrix();
                    
                    this.cubeProxy.matrix.toArray(matrices, index * 16); // Batch the matrix
                    index++;
                }
            }
        }

        this.mesh.instanceMatrix.set(matrices); // Set all matrices at once
        this.mesh.instanceMatrix.needsUpdate = true;
        this.mesh.setColorAt(0, new THREE.Color(0, 0, 0)); // Initial color
        this.mesh.instanceColor.needsUpdate = true;
    }

    updateScalar(indices, scalar) {
        const matrices = new Float32Array(indices.length * 16); // Batch the matrices

        for (let index = 0; index < indices.length; index++) {
            let local = indices[index];
            let meshIndex = local[2] + this.size[0] * local[1] + local[0] * this.size[0] * this.size[1];
            local = local.map((x, i) => x - this.offset[i]);

            this.cubeProxy.position.set(local[0], local[1], local[2]);
            this.cubeProxy.scale.setScalar(scalar);
            this.cubeProxy.updateMatrix();

            this.cubeProxy.matrix.toArray(matrices, index * 16);
            this.mesh.setMatrixAt(meshIndex, this.cubeProxy.matrix);
        }

        this.mesh.instanceMatrix.needsUpdate = true; // Batch update
    }

    updateValues(indices, values) {
        const colors = new Float32Array(indices.length * 3); // Batch the colors

        for (let index = 0; index < indices.length; index++) {
            const local = indices[index];
            let meshIndex = local[2] + this.size[0] * local[1] + local[0] * this.size[0] * this.size[1];
            const value = values[index];
            const color = this.colorbar(value);

            color.toArray(colors, index * 3);
            this.mesh.setColorAt(meshIndex, color); // Update the color at the mesh index
        }

        this.mesh.instanceColor.needsUpdate = true; // Batch update colors
    }

    async update(init = false) {
        if (!this.lenia.isRunning && !init) return;

        const gridVoxelNew = this.lenia.tensor.edges();
        const diffGrid = tf.sub(gridVoxelNew, this.lenia.tensor.gridVoxel);

        const gridVoxelDel = tf.equal(diffGrid, -1);
        const voxelIndicesDel = await tf.whereAsync(gridVoxelDel);
        const voxelIndicesDelArr = voxelIndicesDel.arraySync();

        const newGrid = tf.equal(diffGrid, 1);

        const voxelIndicesNew = await tf.whereAsync(newGrid);
        const voxelIndicesNewArr = voxelIndicesNew.arraySync();

        const currGrid = tf.equal(diffGrid, 0);
        const voxelIndices = await tf.whereAsync(currGrid);
        const voxelIndicesArr = voxelIndices.arraySync();

        const voxelValues = tf.gatherND(this.lenia.tensor.grid, voxelIndices);
        const voxelValuesArr = voxelValues.arraySync();

        this.updateScalar(voxelIndicesNewArr, 1);
        this.updateScalar(voxelIndicesDelArr, 0);
        this.updateValues(voxelIndicesArr, voxelValuesArr);

        this.lenia.tensor.gridVoxel.assign(gridVoxelNew);
        this.onChange();
        tf.dispose([
            voxelIndicesDel,
            voxelIndicesNew,
            voxelIndices,
            voxelValues,
            diffGrid,
            gridVoxelNew,
            gridVoxelDel,
            newGrid,
            currGrid,
        ]);
    }

    colorbar(value) {
        const color = new THREE.Color();
        const colours = this.lenia.colormap(value);
        color.setRGB(colours[0], colours[1], colours[2]);
        return color;
    }
    generateBounds(){
        const size = this.lenia.size.slice()
        const box = new THREE.BoxGeometry(1,1,1);
        const material = new THREE.MeshBasicMaterial({
            color: 0xFFFF00, 
            transparent: true, // Enable transparency
            opacity: 0.5      // Adjust opacity (0 is fully transparent, 1 is fully opaque)
          });
          this.boundingbox = new THREE.Mesh(box, material);
        
    }
    renderBounds(value,direction) {
        if (!this.scene.children.includes(this.boundingbox)){this.scene.add(this.boundingbox);}
        const size = this.lenia.size.map((dim, i) => (i === direction ? value : dim));
        this.boundingbox.scale.set(...size);
        this.boundingbox.needsUpdate = true;
    }
    removeBounds() {
        this.scene.remove(this.boundingbox);
    }
    updateGridSize() {
        this.remove();
        this.init();
    }

    remove(){
        this.canvas.style.display = 'none';
        this.scene.remove(this.mesh);
        this.renderer.dispose();
        this.controls.dispose();
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.mesh = null;
        this.scene = null;
    }
}

class array {
    constructor(lenia) {
        this.lenia = lenia;
        this.canvas = document.getElementById('canvas-2d');
        this.ctx = this.canvas.getContext('2d');
        this.renderBoundsPotential = null;
        this.renderBoundsDirection = null;
        this.shouldRenderBounds = false;
        this.cellSize= 16;
        window.addEventListener('resize', () => this.update());
    }

    render() {
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.canvas.width = this.cellRows * this.cellSize
        this.canvas.height = this.cellCols * this.cellSize
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.width);


        if (this.shouldRenderBounds) {
            this.renderBounds();
        }

        let rowStart = null, rowEnd = null, colStart = null, colEnd = null;

        if (this.shouldRenderBounds){
            rowStart =  this.offset[0];
            rowEnd = this.x + this.offset[0];
            colStart = this.offset[1];
            colEnd = this.y + this.offset[1];
        }
        else{ rowStart = 0; rowEnd = this.cellRows; colStart = 0; colEnd = this.cellCols; }


        // Loop to render cells
        for (let i =  rowStart; i < rowEnd; i++) {
            for (let j = colStart; j < colEnd; j++) {
                const value = this.values[i * this.cellCols + j]; // Get value from cached array
                const color = this.lenia.colormap(value);
                const isVisible = value > 1e-18; // Example condition for visibility
                
                // Set the fill style; if not visible, set to transparent
                if (isVisible) {
                    this.ctx.fillStyle = `rgba(${color[0] * 255}, ${color[1] * 255}, ${color[2] * 255}, 1)`;
                } else {
                    this.ctx.fillStyle = `hsl(0,0%,7%)`;
                }
                this.ctx.fillRect(i * this.cellSize, j * this.cellSize, this.cellSize, this.cellSize);
            }
        }
    }
    init(){
        this.canvas.style.display = 'block';
        this.x = this.lenia.size[2];
        this.y = this.lenia.size[1];

       
    }
    
    update() {
        this.x = this.lenia.size[2];
        this.y = this.lenia.size[1];
        /* this.cellSize = Math.min(this.canvas.clientHeight, this.canvas.clientWidth) / Math.max(this.x, this.y)   /*/
        let relativeSize = null
        if (this.shouldRenderBounds){relativeSize = Math.min(this.canvas.clientHeight, this.canvas.clientWidth) / 128}
        else {relativeSize = Math.min(this.canvas.clientHeight, this.canvas.clientWidth) / Math.max(this.x, this.y);}
        this.cellRows = Math.floor(this.canvas.clientWidth / relativeSize);
        this.cellCols = Math.floor(this.canvas.clientHeight / relativeSize);
    

        this.offset = [Math.floor((this.cellRows - this.x) / 2), Math.floor((this.cellCols - this.y) / 2)]
        const arrayIndices = Array.from({ length: this.cellRows * this.cellCols }, (_, index) => {
            const i = Math.floor(index / this.cellCols);
            const j = index % this.cellCols;
            return [(i+this.offset[0]) % this.y, (j+this.offset[1]) % this.x]; // Apply modulus to fit within array bounds
        });
        this.values = tf.tidy(() => {
            let array = this.lenia.tensor.Condense();
            const tfArrayIndices = tf.tensor(arrayIndices, null, 'int32');
            const values = tf.gatherND(array, tfArrayIndices);
            const valuesArray = values.arraySync();
            return valuesArray;
        });
        this.render();
    }
    remove() {
        this.canvas.style.display = 'none';
        this.ctx.clearRect(0, 0, this.canvas.clientWidth/this.scale, this.canvas.clientHeight/this.scale);
    }
    renderBounds() {
        this.ctx.fillStyle = 'rgba(242, 242, 242, 0.25)';
        const size = this.lenia.size.map((dim, i) => (i === this.renderBoundsDirection ? this.renderBoundsPotential : dim));
        const offset = [Math.floor((this.cellRows - size[2]) / 2), Math.floor((this.cellCols - size[1]) / 2)];
        this.ctx.fillRect(offset[0] * this.cellSize, offset[1] * this.cellSize, size[2] * this.cellSize, size[1] * this.cellSize);
        this.ctx.strokeStyle = 'rgba(242, 242, 242, 0.75)';
        this.ctx.lineWidth = this.cellsize*2;
        this.ctx.strokeRect(this.offset[0] * this.cellSize, this.offset[1] * this.cellSize, this.x * this.cellSize, this.y * this.cellSize);


    }
    removeBounds() {
        this.shouldRenderBounds = false;
        this.renderBoundsDirection = null;
        this.renderBoundsPotential = null;
        this.update();
    }
    updateGridSize() {
        this.removeBounds();
        this.update();
        this.x = this.lenia.size[2];
        this.y = this.lenia.size[1];
    }
}

class UI{
    constructor(lenia){
    this.main = document.getElementById('main');
    this.menu = document.getElementById('menu');
    this.playcontrol = document.getElementById('playcontrol');
    this.sidebarmenu = document.querySelector('.sidebarmenu');
    this.gridparamscontainer = document.getElementById("dimensions-container");
    this.lenia = lenia
    this.gencounter = document.getElementById("gen");
    this.timecounter = document.getElementById("time");
    this.rrange = [0,100,100]
    this.seedcontainer = document.getElementById("seedcont");
    this.seeddisplay = document.getElementById("seed");
    this.namecontainer = document.getElementById("namecont");
    this.namedisplay = document.getElementById("name");
    this.typedisplay = document.getElementById("type");
    this.menuItems = document.querySelectorAll('.menu li a');
    this.containers = document.querySelectorAll('.container');

    this.playButtons = document.querySelectorAll('#play-button-checkbox');
    this.playButtons.forEach((button) => {
        button.checked = false; // Initialize all buttons as unchecked
    });
    
    
    this.isbuttonchecked = false;
    
    this.playButtons.forEach((button) => {
        button.addEventListener('change', () => { // Use 'change' instead of 'check'
            // Check or uncheck all buttons based on the state of the clicked button
            const allChecked = button.checked;
    
            // Set all buttons to the state of the clicked button
            this.playButtons.forEach((btn) => {
                btn.checked = allChecked; 
            });
    
            // Run the appropriate function based on whether a button is checked or unchecked
            if (allChecked && !this.isbuttonchecked) {
                this.isbuttonchecked = true;
                this.lenia.startstate().bind(this.lenia);
            } else if (!allChecked && this.isbuttonchecked) {
                this.isbuttonchecked = false;
                this.lenia.stopstate().bind(this.lenia);
            }
        });
    });

    this.togglesidebarButton = document.getElementById('toggleSidebar')
    this.togglelayoutButton = document.getElementById('toggleLayout')


    this.reset = document.getElementById("reset");
    this.gridsizecounter = document.getElementById("gridsize");
    this.radiuscounter = document.getElementById("radius");
    this.mucounter = document.getElementById("mu");
    this.sigmacounter = document.getElementById("sigma");
    this.seedinput = document.getElementById("seedinput");
    this.muslider = document.getElementById("mu-slider");
    this.sigmaslider = document.getElementById("sigma-slider");
    this.seedWarning = document.getElementById("seedwarning");
    this.dimcounter = document.getElementById("dim");
    this.bcontainer = document.getElementById("bcontainer");
    this.bdimcounter = document.getElementById("bdim");
    this.bdimaddbutton = document.getElementById("bdim-add-button");
    this.bdimsubbutton = document.getElementById("bdim-sub-button");
    this.cdimcounter = document.getElementById("cdim");
    this.clustercontainer  = document.getElementById("clusterparams");
    this.clusterinputmin = document.getElementById("cluster-min-value");
    this.clusterinputmax = document.getElementById("cluster-max-value");
    this.densityslider = document.getElementById("density-slider");
    this.densitycounter = document.getElementById("density");
    
    this.renderdimensionbutton = document.getElementById("render-dimension-button");
    this.renderdimensionbutton.textContent = this.lenia.render;
    this.renderdimensionbutton.addEventListener("click", () => { this.togglerRenderDimension() });  

    this.seedinput.addEventListener("input" , () => {
        this.seedinput.value = this.validateData(this.seedinput.value,'int',16)
    });



    this.densityslider.addEventListener("change", () => {
        const Value = parseFloat(this.densityslider.value)
        this.lenia.cluster_density = Value
        this.updatedensity();
    });
    this.densityslider.addEventListener("input", () => {
        const sliderValue = parseFloat(this.densityslider.value)  ;
        this.densitycounter.value = sliderValue;
        });
    this.densitycounter.addEventListener("blur", () => {
        Value = parseFloat(this.densitycounter.value);
        this.lenia.cluster_density = Value;
        this.updatedensity();
        });

    this.clusterinputmin.addEventListener("change", () => {
        const Value = parseFloat(this.clusterinputmin.value)
        this.clusterValueMin = Value
        $("#cluster-slider-range").slider("values", 0, Value);
    });
    this.clusterinputmax.addEventListener("change", () => {
        const Value = parseFloat(this.clusterinputmax.value)
        this.clusterValueMax = Value
        $("#cluster-slider-range").slider("values", 1, Value);
    });

    this.muslider.addEventListener("change", () => {
        const Value = parseFloat(this.muslider.value) 
        this.lenia.params['m']  = Value
    });
    this.muslider.addEventListener("input", () => {
        const sliderValue = parseFloat(this.muslider.value)  ;
        this.mucounter.value = sliderValue;
      });
      this.mucounter.addEventListener("blur", () => {
        const Value = parseFloat(this.mu.value);
        this.muslider.value = Value;
        this.lenia.params['m']  = Value;
      });
      
      this.sigmaslider.addEventListener("change", () => {
        const Value = parseFloat(this.sigmaslider.value) 
        this.lenia.params['m']  = Value
    });
    this.sigmaslider.addEventListener("input", () => {
        const sliderValue = parseFloat(this.sigmaslider.value)  ;
        this.sigmacounter.value = sliderValue;
      });
    this.sigmacounter.addEventListener("blur", () => {
    const Value = parseFloat(this.sigmacounter.value);
    this.sigmaslider.value = Value;
    this.lenia.params['m']  = Value;
    });

    this.togglelayoutButton.addEventListener("click", () => {
        this.toggleLayout();
    });
    this.togglesidebarButton.addEventListener("click", () => {
        this.toggleSidebar();
    });

    this.transitions = {
        'open': { nextState: 'reduced', width: '140px' },
        'reduced': { nextState: 'collapsed', width: '41px' },
        'collapsed': { nextState: 'open', width: '0px' }
    };

    this.sidebarState = '';
    this.layoutState = '';

    this.aspectRatio = window.innerWidth / window.innerHeight;
    this.sidebarState = 'reduced'
    this.layoutStateAuto = this.aspectRatio > 0.75 ? 'row' : 'column';
    this.layoutState = this.layoutStateAuto;
    this.updateLayoutDimensions();
    this.sidebarmenu.classList.add(this.sidebarState, 'initialised');
    this.main.style.left = this.transitions[this.sidebarState].width;
    this.main.classList.add('initialised');


    
    window.addEventListener('resize', () => this.updateLayoutDimensions());
    window.addEventListener('fullscreenchange', () => this.updateLayoutDimensions());
    window.addEventListener('mozfullscreenchange', () => this.updateLayoutDimensions());
    window.addEventListener('webkitfullscreenchange', () => this.updateLayoutDimensions());
    window.addEventListener('msfullscreenchange', () => this.updateLayoutDimensions());


    // Add event listener for sidebar collapse button with a timeout
    document.querySelector('.sidebar-collapse-button').addEventListener('click', () => {
        setTimeout(this.updateLayoutDimensions, 300);
    });
    this.reset.addEventListener("click", this.lenia.reset.bind(this.lenia))
    this.PopulateAnimalList()
    this.populateParameters(this.gridparamscontainer,["X","Y","Z"],this.lenia.size,[0,128,1],"parameter-row dimension-row",null)
    this.generateClusterRange()
    this.updatedensity()

    this.bcontainer.innerHTML = ''
    const range  = Array.from({ length: this.lenia.params['b'].length + 1 }, (_, index) => index);
    this.populateParameters(this.bcontainer,range,this.lenia.params['b'],[0,1,0.001],"parameter-row beta-row",null)
    this.bdimcounter.textContent = this.lenia.params['b'].length;


    this.tbuttons =  Array.from(document.querySelectorAll('.time-bar .button'));
    this.timeres = document.getElementById('timeres');

}   
editMenu(container, children) {

    if (!container || !container.children) {
        console.error('Invalid container element');
        return;
    }
    // Convert container's children (HTMLCo llection) to an array
    const childrenArray = Array.from(container.children);

    // Hide all the child elements
    childrenArray.forEach(child => {
        child.style.display = 'none';
    });

    // Show only the elements whose IDs are in the children array
    children.forEach(childId => {
        const matchingChild = container.querySelector(`#${childId}`);
        if (matchingChild) {
            matchingChild.style.display = 'block';
        } else {
            console.warn(`Child with id "${childId}" not found in container.`);
        }
    });
}

    updatedensity(){
        this.densityslider.value = this.lenia.cluster_density
        this.densitycounter.value = this.lenia.cluster_density
    }
    generateClusterRange() {
        $("#cluster-min-value").val(this.lenia.cluster_range[0]);
        $("#cluster-max-value").val(this.lenia.cluster_range[1]);
        $("#cluster-slider-range").slider({
        range: true,
        min: 0,
        max: 1,
        step: 0.001,
        values: this.lenia.cluster_range,
        slide: (event, ui) => {
            this.lenia.cluster_range = ui.values
            $("#cluster-min-value").val(this.lenia.cluster_range[0]);
            $("#cluster-max-value").val(this.lenia.cluster_range[1]);
        }
        });

    }

    toggleSidebar() {
        const nextInfo = this.transitions[this.sidebarState];
        this.sidebarmenu.classList.replace(this.sidebarState, nextInfo.nextState);
        this.sidebarState = nextInfo.nextState;
        this.main.style.left = this.transitions[this.sidebarState].width;
        this.updateLayoutDimensions();
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
        const sidebarWidth = parseInt(this.transitions[this.sidebarState].width, 10);
        const mainHeight = window.innerHeight - 32;
        const mainWidth = window.innerWidth - sidebarWidth;

        this.main.style.height = `${mainHeight}px`;
        this.main.style.width = `${mainWidth}px`;
        if (this.layoutState === 'row') {
            this.rowLayout();
        } else {
            this.main.style.height = `auto`;
            this.columnLayout();
        }

    }

    togglerRenderDimension() {
        this.lenia.renderRemove();
        this.lenia.render = (this.lenia.render === '2D') ? '3D' : '2D';
        this.lenia.renderInit();
        this.renderdimensionbutton.textContent = this.lenia.render;
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
    validateCluster(input, numberRange) {
        const isMin = input.id === "cluster-min-value";
        const isMax = input.id === "cluster-max-value";
    
        if (isMin || isMax) {
            const otherInputId = isMin ? "cluster-max-value" : "cluster-min-value";
            const otherInput = document.getElementById(otherInputId);
            const otherValue = otherInput.value ? parseFloat(otherInput.value) : numberRange[isMin ? 1 : 0];
    
            const adjustedRange = isMin
                ? [numberRange[0], Math.min(otherValue, numberRange[1])]
                : [Math.max(numberRange[0], otherValue), numberRange[1]];
            validateInput(input, 'float', adjustedRange);
        }
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
            this.populateParameters(this.gridparamscontainer,["X","Y","Z"],this.lenia.size,[0,128,1],"parameter-row dimension-row",null)
            this.BindParameters(this.gridparamscontainer,this.lenia.size.slice().reverse(),this.lenia.updateGridSize.bind(this.lenia),"change")
            this.BindParameters(this.gridparamscontainer,this.lenia.size.slice().reverse(),this.lenia.renderBounds.bind(this.lenia),"input")
            this.BindParameters(this.gridparamscontainer,this.lenia.size.slice().reverse(),this.lenia.renderRemoveBounds.bind(this.lenia),"mouseup")
            this.BindParameters(this.gridparamscontainer,this.lenia.size.slice().reverse(),this.lenia.renderRemoveBounds.bind(this.lenia),"touchend")

    }   
    updatem() { const value = this.lenia.params['m'];
                this.mucounter.value = value;
                this.muslider.value  = value;}
    updates() {const value = this.lenia.params['s'];
                this.sigmacounter.value = value;
                this.sigmaslider.value  = value;}

    updateb() {const values = this.lenia.params['b'];
                this.bdimcounter.textContent = values.length;
                this.bcontainer.innerHTML = '';
                const range  = Array.from({ length: values.length + 1 }, (_, index) => index);
                this.populateParameters(this.bcontainer,range,values,[0,1,0.001],"parameter-row beta-row",this.lenia.tensor.generateKernel.bind(this.lenia.tensor));
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
    updategen(){this.gencounter.textContent = 'f ' + this.lenia.gen;}
    updateseed() {
        if (window.getComputedStyle(this.seedcontainer).display === "none") {
            this.seedcontainer.style.display = "block";
        }
        if (window.getComputedStyle(this.namecontainer).display === "flex") {
            this.namecontainer.style.display = "none";
        }
        this.seeddisplay.textContent = this.lenia.seed;
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
        
    populateParameters(container, labels, values,ranges,classnames,eventhandler) {
        for (let i = 0; i < values.length; i++) {
            this.Addparameter(container, labels, values, i,ranges,classnames,eventhandler);
        }
      }
      
    Addparameter(container, names, values, index,ranges, classnames,eventhandler) {

        const parameterType = Number.isInteger(ranges[2]) ? 'int' : 'float';
        const allowedChars = parameterType === 'int' ? '0-9' : '0-9.';
        const sliceValue = parameterType === 'int' ? ranges[1].toString().length : 5;

        const row = document.createElement("div");
        classnames.split(" ").forEach(classname => {
            row.classList.add(classname);
        });
      
        const label = document.createElement("label");
        label.classList.add("paras-label");
        label.textContent = names[index].toString()
        row.appendChild(label);
      
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = ranges[0];
        slider.max = ranges[1];
        slider.step = ranges[2];
        slider.value = values[index];
        row.appendChild(slider);
      
        const text = document.createElement("input");
        text.type = "text";
        text.value = values[index];
        text.classList.add("para-input");
        text.addEventListener("input", () => {
            text.value = this.validateData(text.value, parameterType, sliceValue);
        });
        text.style.width = `${sliceValue + 1}ch`;
        text.addEventListener("blur", () => {this.validateInput(text, parameterType, ranges);});
        row.appendChild(text);
      
        // Associate slider and text input using data attributes
      
        // Event listener for both slider and text input
        if (eventhandler) {this.BindParameter(row, values, index, eventhandler)}

        slider.addEventListener("input", (event) => {
            const sliderValue = parseFloat(event.target.value);
            text.value = sliderValue;
          });
      
        container.appendChild(row);
      }
    Removeparameter(container,i){
        container.removeChild(container.childNodes[i])
    }
    BindParameters(container, values, eventhandler,eventSource) {
        for (let i = 0; i < values.length; i++) {
            const rows = container.querySelectorAll('.parameter-row');
            let row = rows[i]; // Get the correct row by index

            this.BindParameter(row, values,i, eventhandler,eventSource); // Bind each parameter in the container
        }
    }
    BindParameter(row, values, index, eventhandler,eventSource) {
        console.log('eventSource',eventSource)
        const slider = row.querySelector("input[type='range']"); // Get the slider input
        const text = row.querySelector(".para-input"); // Get the text input
        const updateValue = (event) => { 
            this.validateInput(event.target, 'int', [slider.min, slider.max]);
            const inputValue = parseFloat(event.target.value);  
            let parameters = [inputValue, index,values];
            parameters = parameters.slice(0, eventhandler.length);  
            if (typeof eventhandler === 'function') {
                eventhandler(...parameters)
            }
        };
        // Bind event handler to both slider and text input

        slider.addEventListener(eventSource, updateValue);
        text.addEventListener(eventSource, updateValue);
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
        this.lenia.renderUpdate()
        this.lenia.UI.updateparams()
        this.lenia.UI.updatename()
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

const lenia = new Lenia();
if (animalArr !== null) {
    lenia.tensor.load.SelectAnimalID(40);
  } else {
    lenia.tensor.randomGrid()
    lenia.tensor.generateKernel();
  }
lenia.renderUpdate(true);



  
