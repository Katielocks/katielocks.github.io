tf.setBackend('webgl')
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );
camera.position.x = 0.8*64 ;
camera.position.y = 0.8*64 ;
camera.position.z = 0.8*64 ;
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; 
controls.dampingFactor = 0.05; // Set the damping factor for the damping effect
scene.background = new THREE.Color(0x101214);



const sliders = document.querySelectorAll('.slider');

  function updateSliderTrack(event) {
    const slider = event.target;
    const value = (slider.value - slider.min) / (slider.max - slider.min);

    // Calculate the width of the track to the left of the thumb
    const trackWidth = value * 100;

    // Set the background color for the left part of the track
    const colorLeft = `linear-gradient(90deg, #00ff00 ${trackWidth}%, #555353 ${trackWidth}% 100%)`;

    // Update the slider's pseudo-element style with the new background color
    slider.style.setProperty('--track-color-left', colorLeft);
  }

  sliders.forEach(slider => {
    // Initially set the background color based on the slider's initial value
    updateSliderTrack({ target: slider });

    // Add an event listener to each slider to update the background color when the value changes
    slider.addEventListener('input', updateSliderTrack);
  });




const animate = function () {
    requestAnimationFrame( animate );
    renderer.render( scene, camera );
  };
class Lenia {
    constructor() {
        this.params = {'R':15, 'T':10, 'b':[1], 'm':0.1, 's':0.01, 'kn':1, 'gn':1};
        this.DIM = 3;
        this.size = Array(this.DIM).fill(64);
        this.cluster_size = this.size.map(x => x);
        this.cluster_density = 1
        this.cluster_range = [0,1]
        this.seed = null
        this.id = null
        this.isRunning = false;
        this.time = 0;
        this.gen = 0;
        this.drawcall = renderer.info.render.calls;
        this.tensor = new tensor(this);
        this.mesh = new mesh(this);
        this.UI = new UI(this);

        }
        animatestate() {
            if (this.isRunning) {
              // Store the current instance of "this"
              const self = this;
              // Update the grid every 1/T seconds
              setTimeout(function() {
                self.tensor.update();
                const a = self.mesh.update();
                self.animatestate();
                this.gen += 1
                this.time += 1/this.params['T']
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
        reset() {
            if (this.id !== null) {
                this.tensor.load.SelectAnimalID(this.id)
            }
            else{
                this.tensor.randomGrid()
                this.mesh.update(true)
            }
            this.time = 0;
            this.gen = 0;
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
        this.lenia.mesh.update(true)
        new_grid.dispose()
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
    updateGridSize() {
        this.lenia.mesh.size = this.lenia.size.slice(-3)
        this.lenia.mesh.offset = this.lenia.mesh.size.map(x=>(x-1)/2)
        tf.dispose([this.gridVoxel, this.ekernel, this.kernel]);
        const dummy = this.resize(this.grid, this.lenia.size);
        this.grid.dispose()
        this.grid = tf.variable(dummy);
        this.init();
        scene.remove(this.lenia.mesh.mesh);
        this.lenia.mesh.init();
        scene.add(this.lenia.mesh.mesh);
        this.lenia.mesh.update(true); // Pass the grid as a parameter


    }
}

class mesh {
    constructor(lenia){
        this.lenia = lenia
        this.cubeProxy = new THREE.Object3D()
        this.size = this.lenia.size.slice(-3)
        const error = this.size.map(x=>Math.abs(x%2))
        this.offset = this.size.map(x=>(x-1)/2)
        this.init()
        this.colourbarmax = [253/255,232/255,64/255]
        this.colourbarmin = [68/255,13/255,84/255]

    }
    init() {
        this.mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshBasicMaterial(), this.size.reduce((a, v) => a * v))
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
        const value = values[index]
        const colour = this.colorbar(value)
        this.mesh.setColorAt(meshindex, colour);

    }}
    async update(init=false) {
        if (this.lenia.isRunning === false && init === false) {return}
        const gridVoxel_new = this.lenia.tensor.edges()
        const diffgrid = tf.sub(gridVoxel_new,this.lenia.tensor.gridVoxel)

        const gridVoxel_del = tf.equal(diffgrid,-1)
        const voxelIndices_del = await tf.whereAsync(gridVoxel_del)
        const voxelIndices_del_arr= voxelIndices_del.arraySync()

        const newgrid = tf.equal(diffgrid,1)
        const voxelIndices_new = await tf.whereAsync(newgrid)
        const voxelIndices_new_arr = voxelIndices_new.arraySync()
        const currgrid = tf.equal(diffgrid,0)
        const voxelIndices = await tf.whereAsync(currgrid)
        const voxelIndices_arr = voxelIndices.arraySync()

        const voxelvalues = tf.gatherND(this.lenia.tensor.grid, voxelIndices)
        const voxelvalues_arr = voxelvalues.arraySync()
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
          currgrid
        ]);
      }
      colorbar(value){
        const color = new THREE.Color()
        const colours = this.colourbarmax.map((x,i)=>(x-this.colourbarmin[i])*value+this.colourbarmin[i])
        color.setRGB(colours[0],colours[1],colours[2])
        return color
      }

    

}

class UI{
    constructor(lenia){
    this.gridparamscontainer = document.getElementById("gridparams");
    this.lenia = lenia
    this.gencounter = document.getElementById("gen");
    this.timecounter = document.getElementById("time");
    this.drawcallcounter = document.getElementById("drawcall");
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
    this.radiusAddButton = document.getElementById("radius-add-button");
    this.radiusSubButton = document.getElementById("radius-sub-button");
    this.timeAddButton = document.getElementById("time-add-button");
    this.timeSubButton = document.getElementById("time-sub-button");
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
    this.generatebutton = document.getElementById("generate-button");
    this.colorbarmin = document.getElementById("colorbar-min");
    this.colorbarmax = document.getElementById("colorbar-max");

    this.colorbarmin.addEventListener("blur", () => {this.updateColorbar()})
    this.colorbarmax.addEventListener("blur", () => {this.updateColorbar()})
    this.colorbarmin.value =  this.RGBtoHex(this.lenia.mesh.colourbarmin)
    this.colorbarmax.value =  this.RGBtoHex(this.lenia.mesh.colourbarmax)

    this.generatebutton.addEventListener("click", () => {
        if (this.seedinput.value.length === 8) {
        this.lenia.seed  = parseInt(this.seedinput.value);}
        else if (this.seedinput.value.length !== 8 && this.seedinput.value.length !== 0) {return}
        if (this.seedinput.value.length === 0){this.lenia.tensor.generateseed()}
        this.lenia.tensor.randomGrid();
        this.lenia.mesh.update(true);
        this.lenia.id = null;
        this.updateseed()
    })

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
    
    this.bdimaddbutton.addEventListener("click", () => {
        this.lenia.params['b'].push(0)
        this.updateb();
    });
    this.bdimsubbutton.addEventListener("click", () => {
        this.lenia.params['b'].pop()
        this.updateb();})


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


    this.reset.addEventListener("click", this.lenia.reset.bind(this.lenia))
    this.updatedim()
    this.PopulateAnimalList()
    this.updateparams()
    this.populateParameters(this.gridparamscontainer,["x","y","z"],this.lenia.size,[0,100,1],this.lenia.tensor.updateGridSize.bind(this.lenia.tensor))
    this.populateParameters(this.clustercontainer,["x","y","z"],this.lenia.cluster_size,[0,100,1],null)
    this.generateClusterRange()
    this.updatedensity()
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
    updatedim(){this.dimcounter.textContent = this.lenia.DIM;} 
    updateb() {
        this.lenia.tensor.generateKernel()
        this.bcontainer.innerHTML = ''
        const range  = Array.from({ length: this.lenia.params['b'].length + 1 }, (_, index) => index);
        this.populateParameters(this.bcontainer,range,this.lenia.params['b'],[0,1,0.001],this.lenia.tensor.generateKernel.bind(this.lenia.tensor))
        this.bdimcounter.textContent = this.lenia.params['b'].length;
      }
    updatem() { const value = this.lenia.params['m'];
                this.mucounter.value = value;
                this.muslider.value  = value;}
    updates() {const value = this.lenia.params['s'];
                this.sigmacounter.value = value;
                this.sigmaslider.value  = value;}
    updateparams() {
        this.updateradius();
        this.updatet();
        this.updatem();
        this.updates();
        this.updateb();
    }
    updatetime(){this.timecounter.textContent = this.lenia.time.toFixed(2);}
    updategen(){this.gencounter.textContent = this.lenia.gen;}
    updatedrawcall(){this.drawcallcounter.textContent = this.lenia.drawcall;}
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
        this.lenia.mesh.colourbarmin = this.HextoRGB(this.colorbarmin.value)
        this.lenia.mesh.colourbarmax = this.HextoRGB(this.colorbarmax.value)
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
                    const code = li.appendChild(document.createElement("DIV"));
                    text.title = a[0] + " " + engSt.join(" ") + "\n" + ruleSt;
                    if (sameCode) codeSt = " ".repeat(codeSt.length);
                    if (sameEng0) engSt[0] = lastEng0.substring(0, 1) + ".";
                    text.innerHTML = engSt.join(" ");
                    li.style.color = "#cdd0d6"
                    li.style.width = "90%"
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
        
    populateParameters(container, labels, values,ranges, eventhandler) {
        for (let i = 0; i < values.length; i++) {
            this.Addparameter(container, labels, values, i,ranges, eventhandler);
        }
      }
      
    Addparameter(container, names, values, index,ranges, eventhandler) {
        const row = document.createElement("div");
        row.classList.add("parameter-row");
      
        const label = document.createElement("label");
        label.textContent = names[index].toString() + ": ";
        row.appendChild(label);
      
        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = ranges[0];
        slider.max = ranges[1];
        slider.step = ranges[2];
        slider.value = values[index];
        slider.style.width = "80%";
        row.appendChild(slider);
      
        const text = document.createElement("input");
        text.type = "text";
        text.value = values[index];
        text.style.width = "10%";
        text.classList.add("seed-input");
        row.appendChild(text);
      
        // Associate slider and text input using data attributes
      
        // Event listener for both slider and text input
        const updateValue = (event) => {
          const inputValue = parseFloat(event.target.value);
          values[index] = inputValue;
          if (typeof eventhandler === 'function') {
            eventhandler(); 
          }
        };
        slider.addEventListener("input", (event) => {
            const sliderValue = parseFloat(event.target.value);
            text.value = sliderValue;
          });
        slider.addEventListener("change", updateValue);
        text.addEventListener("blur", updateValue);
      
        container.appendChild(row);
      }
    Removeparameter(container,i){
        container.removeChild(container.childNodes[i])
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
        this.lenia.mesh.update(true);
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
scene.add(lenia.mesh.mesh);
if (animalArr !== null) {
    lenia.tensor.load.SelectAnimalID(40);
  } else {
    lenia.tensor.randomGrid()
    lenia.tensor.generateKernel();
    lenia.mesh.update(true);
  }
animate();



  
