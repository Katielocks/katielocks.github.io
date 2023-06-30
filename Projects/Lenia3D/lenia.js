tf.setBackend('webgl')
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );
camera.position.x = 1.2*64 ;
camera.position.y = 1.2*64 ;
camera.position.z = 1.2*64 ;
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; 
controls.dampingFactor = 0.05; // Set the damping factor for the damping effect
scene.background = new THREE.Color(0x101214);
const animate = function () {
    requestAnimationFrame( animate );
    controls.update(); // Update controls
    renderer.render( scene, camera );
  };
class Lenia {
    constructor() {
        this.grid = null, this.gridVoxel = null, this.kernel = null;
        this.params = {'R':15, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'kn':1, 'gn':1};
        this.DIM = 3;
        this.size = Array(this.DIM).fill(64);
        this.tensor = new tensor(this);
        this.mesh = new mesh(this);
        this.UI = new UI(this);
        this.seed = null
        this.isRunning = false;
        this.time = 0;
        this.gen = 0;

        }
        animatestate() {
            if (this.isRunning) {
              // Store the current instance of "this"
              const self = this;
              // Update the grid every 1/T seconds
              setTimeout(function() {
                const T = Date.now();
                self.tensor.update();
                self.mesh.update();
                self.animatestate();
                this.gen += 1
                this.time += 1/this.params['T']
                console.log(this.params)
                this.UI.updatetime()
                this.UI.updategen()
              }.bind(this), Math.floor(1000 / this.params['T']));
          
              // Render your game objects here
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
                this.UI.startstop.style.backgroundImage = "url('startbutton.png')";
                this.stopstate();
            } else {
                this.UI.startstop.style.backgroundImage = "url('pausebutton.png')";
                this.startstate();
            }
        }


}


class tensor {
    constructor(lenia) {
      this.lenia = lenia;
      this.init();
      this.cluster_size = this.lenia.size.map(x => x);
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
        
    }
    init() {
        this.grid = tf.tidy(() =>  {return tf.variable(tf.zeros(this.lenia.size))});
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
        return this.fftn(this.resize(kern_weights,size).cast('complex64'));

    })}
    transposecomplex(tensor,perms){
        return tf.tidy(()=>{
          const size = perms.map(index => tensor.shape[index])
          let real = tf.real(tensor).reshape(size).transpose(perms)
          let imag = tf.imag(tensor).reshape(size).transpose(perms)
          return tf.complex(real,imag)
        })
    }
      
    fft1(tensor,perms){
        return  tf.tidy(() => {
            let fttensor = tf.spectral.fft(tensor.transpose(perms))
        return this.transposecomplex(fttensor,perms)})
    }
    
    fftn(tensor){
        return tf.tidy(() => {
            let perms = tf.range(0,tensor.shape.length,1).arraySync()
            let fttensor = tf.tidy(() => {return this.fft1(tensor,perms)})
            for (let i=1;i<perms.length;i++){
                const size = perms.length-1
                const permsi = perms.slice();
                [permsi[size],permsi[size-i]] = [permsi[size-i],permsi[size]];
                fttensor = tf.tidy(() => {return this.fft1(fttensor,permsi)})}
            return fttensor
        })
    }
    
    ifft1(tensor,perms){
        return  tf.tidy(() => {
        let fttensor = tf.spectral.ifft(tensor.transpose(perms))
        return this.transposecomplex(fttensor,perms)})
    }
    
    ifftn(tensor){
        return tf.tidy(() => {
            let perms = tf.range(0,tensor.shape.length,1).arraySync()
            let fttensor = tf.tidy(() => {return this.ifft1(tensor,perms)})
            for (let i=1;i<perms.length;i++){
                const size = perms.length-1
                const permsi = perms.slice();
                [permsi[size],permsi[i-1]] = [permsi[i-1],permsi[size]];
                fttensor = tf.tidy(() => {return this.ifft1(fttensor,permsi)})
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
        let padding =  size.map((x,i)=>x-grid.shape[i])
        const error = padding.map(x=>Math.abs(x%2))
        padding = padding.map(x=>Math.floor(x/2))
        padding = padding.map((x,i)=>[x,x+error[i]])
        return tf.pad(grid,padding)
    }
    randomGrid(){
        if (this.lenia.seed == null){this.generateseed()}
        const new_grid = tf.tidy(() => {return tf.variable(this.resize(tf.randomUniform(this.cluster_size, 0, 1, 'float32', seed),this.lenia.size))})
        this.grid.assign(new_grid)
        new_grid.dispose()}
    async resetVoxelGrid() {
        this.randomGrid()
        this.lenia.mesh.update()
        }
    generateseed() {
        this.lenia.seed = Math.floor(Math.random() * Math.pow(10, 8));
    }
    edges() {
        // Define the weights of the directional kernel
        return tf.tidy(() => {
            const padding = [[1, 1], [1, 1], [1, 1]];
            const grid_bool = tf.notEqual(tf.pad(this.grid,padding),0)  
            let egrid = tf.real(this.ifftn(tf.mul(this.fftn(grid_bool.cast('complex64')), this.ekernel)))
            const mid  = this.lenia.size.map(x=>Math.floor(x/2)+2);
            egrid = tf.round(this.roll(egrid, [0,1,2], mid))
            egrid = tf.slice(egrid, [1,1,1], this.lenia.size)
        return  egrid.clipByValue(0,1).cast('int32')})
    }
    
    update() {  
        const dummy  = tf.tidy(() => {
            const fftngrid = this.fftn(this.grid.cast('complex64'))
            const potential_FFT = tf.mul(fftngrid, this.kernel)
            const dt = 1/this.lenia.params['T']
            const U = tf.real(this.ifftn(potential_FFT))
            const gfunc = this.growth_func[this.lenia.params['gn'] - 1]
            const field = gfunc(U, this.lenia.params['m'], this.lenia.params['s'])
            const grid_new = tf.add(this.grid,tf.mul(dt,field))
            return grid_new.clipByValue(0,1)
        })
        this.grid.assign(dummy)
        dummy.dispose()
        }
    zoom3D(Tensor,zoom){
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
}

class mesh {
    constructor(lenia){
        this.lenia = lenia
        this.cubeProxy = new THREE.Object3D()
        this.size = this.lenia.size.slice(-3)
        this.offset = this.size.map(x=>x/2-0.5)
        this.mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial(), this.size.reduce((a, v) => a * v))
        this.initVoxelgrid()

    }
    initVoxelgrid() {
        for (let i = 0; i < this.size[0]; i++) {
            for (let j = 0; j < this.size[1]; j++) {
                for (let k = 0; k < this.size[2]; k++) {
                    const coord = [i,j,k].map((x,index)=>x-this.offset[index])
                    this.cubeProxy.position.set(coord[0], coord[1], coord[2]);
                    this.cubeProxy.scale.setScalar(0)
                    this.cubeProxy.updateMatrix()
                    const index = k+this.size[0]*j+i*this.size[0]*this.size[1]
                    this.mesh.setMatrixAt(index, this.cubeProxy.matrix)
        }   }   }
        this.mesh.setColorAt(0, new THREE.Color(0, 0, 0));

    }
    updatescalar(indices,scalar){
        for (let index = 0; index < indices.length; index++) {
          let local = indices[index]
          let meshindex = local[2]+this.size[0]*local[1]+local[0]*this.size[0]*this.size[1]
          local = local.map((x,i)=>x-this.offset[i])
          this.mesh.getMatrixAt(meshindex, this.cubeProxy.matrix)
          this.cubeProxy.position.set(local[0], local[1], local[2]);
          this.cubeProxy.scale.setScalar(scalar)
          this.cubeProxy.updateMatrix()
          this.mesh.setMatrixAt(meshindex, this.cubeProxy.matrix)
        }
    }
      
    updatevalues(indices,values){
      for (let index = 0; index < indices.length; index++) {
        const local = indices[index]
        const shape = this.lenia.size.slice(-3)
        let meshindex = local[2]+shape[0]*local[1]+local[0]*shape[0]*shape[1]
        this.mesh.setColorAt(meshindex, new THREE.Color(0, values[index], 0));

    }}
    async update() {
        let gridVoxel_new = this.lenia.tensor.edges()
        const diffgrid = tf.sub(gridVoxel_new,this.lenia.tensor.gridVoxel)

        const gridVoxel_del = tf.equal(diffgrid,-1)
        let voxelIndices_del = await tf.whereAsync(gridVoxel_del)
        let voxelIndices_del_arr= voxelIndices_del.arraySync()

        const newgrid = tf.equal(diffgrid,1)
        let voxelIndices_new = await tf.whereAsync(newgrid)
        let voxelIndices_new_arr = voxelIndices_new.arraySync()
        let voxelIndices = await tf.whereAsync(tf.notEqual(gridVoxel_new,0))
        let voxelIndices_arr = voxelIndices.arraySync()
        const voxelvalues = tf.gatherND(this.lenia.tensor.grid, voxelIndices)
        let voxelvalues_arr = voxelvalues.arraySync()
        this.updatescalar(voxelIndices_new_arr,1)
        this.updatescalar(voxelIndices_del_arr,0)
        this.updatevalues(voxelIndices_arr,voxelvalues_arr)
        this.mesh.instanceMatrix.needsUpdate = true;
        this.mesh.instanceColor.needsUpdate = true;
        this.lenia.tensor.gridVoxel.assign(gridVoxel_new)
        tf.dispose([
          voxelIndices_del,
          voxelIndices_new,
          voxelIndices ,
          voxelvalues,
          diffgrid,
          gridVoxel_new,
          gridVoxel_del,
          newgrid,
        ]);
      }
    

}

class UI{
    constructor(lenia){
    this.lenia = lenia
    this.gencounter = document.getElementById("gen");
    this.timecounter = document.getElementById("time");
    this.rrange = [0,100,100]
    this.trange = [0,20,20]
    this.seedcontainer = document.getElementById("seedcont");
    this.seeddisplay = document.getElementById("seed");
    this.namecontainer = document.getElementById("namecont");
    this.namedisplay = document.getElementById("name");
    this.typedisplay = document.getElementById("type");
    this.menuItems = document.querySelectorAll('.menu li a');
    this.containers = document.querySelectorAll('.container');
    document.addEventListener('DOMContentLoaded', () => this.menuvar());
    this.startstop = document.getElementById("startstop");
    this.reset = document.getElementById("reset");
    this.startstop.addEventListener("click", () => {this.lenia.updatestate()});
    this.gridsizecounter = document.getElementById("gridsize");
    this.radiuscounter = document.getElementById("radius");
    this.timerescounter = document.getElementById("timeres");
    this.mucounter = document.getElementById("mu");
    this.sigmacounter = document.getElementById("sigma");
    this.seedinput = document.getElementById("seedinput");
    this.seedbutton = document.getElementById("seed-button");
    this.gridsizeslider = document.getElementById("gridsize-slider");
    this.radiusAddButton = document.getElementById("radius-add-button");
    this.radiusSubButton = document.getElementById("radius-sub-button");
    this.timeAddButton = document.getElementById("time-add-button");
    this.timeSubButton = document.getElementById("time-sub-button");
    this.muslider = document.getElementById("mu-slider");
    this.sigmaslider = document.getElementById("sigma-slider");
    this.seedWarning = document.getElementById("seedwarning");
    this.gridsizeslider.addEventListener("change", () => {
    new_grid_size = gridsizeslider.value
    this.updategridsize();
      });
    this.radiusAddButton.addEventListener("click", () => {
    this.lenia.params['R'] = this.inc(this.lenia.params['R'], this.rrange,1)
    this.updateradius();
    });
    
    this.radiusSubButton.addEventListener("click", () => {
    this.lenia.params['R'] = this.inc(this.lenia.params['R'], this.rrange,-1)
    this.updateradius();
    });
    
    this.timeAddButton.addEventListener("click", () => {
    this.lenia.params['T'] = this.inc(this.lenia.params['T'], this.trange,1)
    this.updatet();
    });
    
    this.timeSubButton.addEventListener("click", () => {
    this.lenia.params['T'] = this.inc(this.lenia.params['T'], this.trange,-1)
    this.updatet();
    });
    
    this.muslider.addEventListener("change", () => {
    this.lenia.params['m']  = parseFloat(this.muslider.value) 
    this.updatem();
    });
    
    this.sigmaslider.addEventListener("change", () => {
    this.lenia.params['s']  = parseFloat(this.sigmaslider.value)
    this.updates();
    });
    this.seedbutton.addEventListener("click", () => {
    this.lenia.seed = parseInt(this.seedinput.value)
    this.updateseed();
    this.lenia.tensor.randomGrid()
    this.lenia.gen = 0
    this.lenia.time = 0
    this.lenia.mesh.update()

    });
    this.PopulateAnimalList()
    this.updateparams()
    }
    inc(value, increments,dir) {
    const range = increments[1] - increments[0]; // Calculate the range
    const step = dir*range / increments[2]; // Calculate the step size
        const nextValue = value + step; // Calculate the next value
    // Check if the next value exceeds the maximum
    if (nextValue > increments[1] || nextValue < increments[0]) {
    return value;
    }
    
    return nextValue;
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
    updatet() {this.timerescounter.textContent = this.lenia.params['T'].toFixed(0);}
    updatem() {this.mucounter.textContent = this.lenia.params['m']}
    updates() {this.sigmacounter.textContent = this.lenia.params['s']}
    updateparams() {
        this.updateradius();
        this.updatet();
        this.updatem();
        this.updates();
    }
    updatetime(){this.timecounter.textContent = this.lenia.time.toFixed(2);}
    updategen(){this.gencounter.textContent = this.lenia.gen;}

    updateseed() {
        if (this.seedinput.value.length  === 8) {;
        if (window.getComputedStyle(this.seedcontainer).display === "none") {
            this.seedcontainer.style.display = "block";
        }
        if (window.getComputedStyle(this.namecontainer).display === "flex") {
            this.namecontainer.style.display = "none";
        }
        this.lenia.seed  = parseInt(this.seedinput.value);
        this.seeddisplay.textContent = this.lenia.seed;
      }
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
                    li.title = a[0] + " " + engSt.join(" ") + "\n" + ruleSt;
                    if (sameCode) codeSt = " ".repeat(codeSt.length);
                    if (sameEng0) engSt[0] = lastEng0.substring(0, 1) + ".";
                    li.innerHTML = codeSt + " " + engSt.join(" ")  + " " + this.GetLayerSt(ruleSt);
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
    validateSeedInput(input) {
        const seedLength = input.value.length;
        // Remove any non-numeric characters from the input value
        input.value = input.value.replace(/\D/g, '');
        input.value = input.value.slice(0, 8);
        if (seedLength < 8 && seedLength > 0) {
          this.seedWarning.style.visibility = 'visible';
        } else {
          this.seedWarning.style.visibility = 'hidden';
        }
      }
    
}


class Load{
    constructor(lenia){
        this.lenia = lenia;
        this.DIM_DELIM = {0:'', 1:'$', 2:'%', 3:'#', 4:'@A', 5:'@B', 6:'@C', 7:'@D', 8:'@E', 9:'@F'}

    }
    SelectAnimalID(id) {
        if (id < 0 || id >= animalArr.length) return;
        var a = animalArr[id];
        if (Object.keys(a).length < 4) return;
        const cellSt = a['cells'];
        let dummy = tf.tidy(() =>{ 
            const newgrid  = this.rle2arr(cellSt);
            return this.lenia.tensor.resize(newgrid,this.lenia.size);
        });
        this.lenia.tensor.grid.assign(dummy);

        this.lenia.UI.name = a['name']
        this.lenia.UI.type = this.SelectType(id)
        this.lenia.params = a['params']
        this.lenia.params['b'] = this.st2fracs(this.lenia.params['b'])
        dummy.dispose();
        this.lenia.mesh.update();
        this.lenia.tensor.generateKernel()
        this.lenia.UI.updateparams()
        this.lenia.UI.updatename()
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
scene.add(lenia.mesh.mesh);
if (animalArr !== null) {
    lenia.tensor.load.SelectAnimalID(40);
  } else {
    lenia.tensor.randomGrid()
    lenia.tensor.generateKernel();
    lenia.mesh.update();
  }
animate();

  