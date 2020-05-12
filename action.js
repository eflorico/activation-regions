/**
 * Linear Regions
 */
function LinearRegion(fn_weight, fn_bias, vertices, disabled_neurons, pattern, edgeIds) {
    this.fn_weight = fn_weight;
    this.fn_bias = fn_bias;
    this.vertices = vertices;
    this.disabled_neurons = disabled_neurons;
    this.pattern = pattern;
    this.edgeIds = edgeIds;
}

function intersectLines(line_weight, line_bias, pt1, pt2) {
    var t = (math.multiply(pt1, line_weight).get([0]) + line_bias) / math.multiply(math.subtract(pt1, pt2), line_weight).get([0]);
    return math.add(pt1, math.multiply(t, math.subtract(pt2, pt1)));
}

LinearRegion.prototype.split = function(new_weight, new_bias, idx, edgeId) {
    // Linear function for this neuron
    var weight = math.multiply(this.fn_weight, new_weight);
    var bias = math.multiply(this.fn_bias, new_weight).get([0]) + new_bias;

    // Preactivations at vertices
    var vertex_images = math.add(math.multiply(this.vertices, weight), bias).toArray();

    // Neuron does not cut region
    if (vertex_images.every(x => x > 0)) {
        return new LinearRegion(this.fn_weight, this.fn_bias, this.vertices, this.disabled_neurons, this.pattern + '1', this.edgeIds);
    } else if (vertex_images.every(x => x <= 0)) {
        return new LinearRegion(this.fn_weight, this.fn_bias, this.vertices, this.disabled_neurons.concat([idx]), this.pattern + '0', this.edgeIds);
    // Neuron does cut region
    } else {
        var all_vertices = this.vertices.toArray();
        var pos_vertices = [], neg_vertices = [];
        var pos_edgeids = [], neg_edgeids = [];

        for (var i = 0; i < all_vertices.length; ++i) {
            var j = (i + 1) % all_vertices.length;
            var v_i = all_vertices[i], v_j = all_vertices[j];

            if (vertex_images[i] > 0) {
                pos_vertices.push(v_i);
                pos_edgeids.push(this.edgeIds[i]);
            } else {
                neg_vertices.push(v_i);
                neg_edgeids.push(this.edgeIds[i]);
            }

            if ((vertex_images[i] > 0) ^ (vertex_images[j] > 0)) {
                var intersection = intersectLines(weight, bias, math.matrix(v_i), math.matrix(v_j));
                pos_vertices.push(intersection);
                neg_vertices.push(intersection);

                if (vertex_images[i] > 0) {
                    pos_edgeids.push(edgeId);
                    neg_edgeids.push(this.edgeIds[i]);
                } else {
                    pos_edgeids.push(this.edgeIds[i]);
                    neg_edgeids.push(edgeId);
                }
            }
        }

        return [
            new LinearRegion(this.fn_weight, this.fn_bias, math.matrix(neg_vertices), this.disabled_neurons.concat([idx]), this.pattern + '0', neg_edgeids),
            new LinearRegion(this.fn_weight, this.fn_bias, math.matrix(pos_vertices), this.disabled_neurons, this.pattern + '1', pos_edgeids)
        ];
    }
};

LinearRegion.prototype.nextLayer = function(new_weight, new_bias) {
    this.fn_weight = math.multiply(this.fn_weight, new_weight);
    this.fn_bias = math.add(math.multiply(this.fn_bias, new_weight), new_bias);

    if (this.disabled_neurons.length > 0) {
        var num_rows = this.fn_weight.size()[0];
        this.fn_weight = this.fn_weight.subset(math.index(math.range(0, num_rows), this.disabled_neurons), math.zeros(num_rows, this.disabled_neurons.length));
        this.fn_bias = this.fn_bias.subset(math.index(this.disabled_neurons), this.disabled_neurons.length > 1 ? math.zeros(this.disabled_neurons.length) : 0);    
    }
    this.disabled_neurons = [];
};

LinearRegion.prototype.getColor = function() {
    var hash = 0, i, chr;
    for (i = 0; i < this.pattern.length; i++) {
      chr   = this.pattern.charCodeAt(i);
      hash  = ((hash << 5) - hash) + chr;
      hash |= 0; // Convert to 32bit integer
    }

    function rng() {
        var x = Math.sin(hash++) * 10000;
        return x - Math.floor(x);    
    }

    return [rng(), rng(), rng()];
};

// Returns list of LinearRegions
function calcAR() {
    var vertices = math.matrix([[-1, -1], [1, -1], [1, 1], [-1, 1]]);
    var regions = [new LinearRegion(math.identity(2), math.zeros(2), vertices, [], '', [])];

    var depth = weights.length;
    // Loop over layers
    for (var k = 0; k < depth; ++k) {
        // Loop over neurons
        var all_rows = math.range(0, weights[k].size()[0]);
        for (var n = 0; n < biases[k].size()[0]; ++n) {
            var new_regions = [];
            for (var region of regions) {
                new_regions = new_regions.concat(region.split(
                    weights[k].subset(math.index(all_rows, n)),
                    biases[k].subset(math.index(n)),
                    n,
                    k + '/' + n
                ));
            }
            regions = new_regions;
        }

        for (var region of regions) {
            region.nextLayer(weights[k], biases[k]);
        }
    }

    return regions;
}

// Calculates activation regions and renders them in 3D
function renderAR() {
    while(scene.children.length > 0){ 
        scene.remove(scene.children[0]); 
    }

    var regions = calcAR();
    
    for (var region of regions) {
        var z = math.multiply(region.vertices, region.fn_weight).map((val, idx) => val + region.fn_bias.get([idx[1]])).toArray();
        var xy = region.vertices.toArray();

        var vertices2d = [];
        var geometry3d = new THREE.Geometry();
        for (var i = 0; i < xy.length; ++i) {
            vertices2d.push(new THREE.Vector2(xy[i][0], xy[i][1]));
            geometry3d.vertices.push(new THREE.Vector3(xy[i][0], xy[i][1], z[i]));
        }

        var triangles = THREE.ShapeUtils.triangulateShape(vertices2d, []);
        for (var triangle of triangles) {
            geometry3d.faces.push(new THREE.Face3(triangle[0], triangle[1], triangle[2]));
        }

        geometry3d.computeFaceNormals();

        var color = region.getColor();
        var material = new THREE.MeshPhongMaterial( { color: new THREE.Color(color[0], color[1], color[2]), side: THREE.DoubleSide } );
        var mesh = new THREE.Mesh(geometry3d, material);
        scene.add(mesh);

        for (var i = 0; i < xy.length - 1; ++i) {
            if (region.edgeIds[i] === highlightedEdge) {
                var lineGeo = new THREE.Geometry().setFromPoints([
                    geometry3d.vertices[i],
                    geometry3d.vertices[i + 1]
                ]);
                var lineMat = new THREE.LineBasicMaterial({color: 'black', linewidth: 3, depthTest: false});
                var line = new THREE.Line(lineGeo, lineMat);
                scene.add(line);
            }
        }
    }

    var light = new THREE.PointLight( 0xffffff, 1, 100 );
    light.position.set( 0, 0, 3 );
    scene.add( light );

    light = new THREE.PointLight( 0xffffff, 1, 100 );
    light.position.set( 0, 0, -3 );
    scene.add( light );

    renderer.render( scene, camera );
}

/**
 * Initialization of nnet
 */

// Generate random number according to Gaussian with unit variance
function randn_bm() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random(); //Converting [0,1) to (0,1)
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function initWeights(weight_initializer, bias_initializer) {
    weights = [];
    biases = [];

    for (var i = 0; i < numNeurons.length; ++i) {
        var numPrevNeurons = i > 0 ? numNeurons[i - 1] : 2;
        
        weights.push(math.zeros(numPrevNeurons, numNeurons[i]).map(() => weight_initializer()));
        biases.push(math.zeros(numNeurons[i]).map(() => bias_initializer()));
    }
}

/**
 * Matrix helpers
 */
function dropAlongDim(mat, idx, axis) {
    var indices = [];
    var outputShape = [];

    for (var ax = 0; ax < mat.size().length; ++ax) {
        var size = mat.size()[ax];
        if (ax == axis) {
            indices.push(math.concat(math.range(0, idx), math.range(idx + 1, size)));
            outputShape.push(size - 1);
        } else {
            indices.push(math.range(0, size));
            outputShape.push(size);
        }
    }

    var output = math.subset(mat, math.index.apply(this, indices));

    if (math.isNumber(output)) {
        output = math.reshape(math.matrix([output]), outputShape);
    }

    return output;
}

/**
 * UI rendering
 */
function renderControls() {
    var html = '';

    // Generate HTML
    for (var i = 0; i < numNeurons.length; ++i) {
        var numPrevNeurons = i > 0 ? numNeurons[i - 1] : 2;
        
        html += '<div class="layer">';
        for (var j = 0; j < numNeurons[i]; ++j) {
            html += '<div class="neuron card">';
            html += '<div class="card-body">';
            for (var k = 0; k < numPrevNeurons; ++k) {
                html += '<div class="slider">';
                html += '<input type="range" min="-2" max="2" step="0.001" value="' + weights[i].get([k, j]) + '" class="weight custom-range">';
                html += '<span>' + Math.round(weights[i].get([k, j]) * 1000) / 1000 + '</span>';
                html += '</div>';
            }
            html += '<hr>';
            html += '<div class="slider">';
            html += '<input type="range" min="-2" max="2" step="0.001" value="' + biases[i].get([j]) + '" class="bias custom-range">';
            html += '<span>' + Math.round(biases[i].get([j]) * 1000) / 1000 + '</span>';
            html += '</div>';
            html += '</div>';
            html += '</div>';
        }

        html += '<div class="btn-group">';
        if (i < numNeurons.length - 1) {
            html += '<button type="button" class="btn btn-primary add-neuron">Add neuron</button>';
        } else {
            html += '<button type="button" class="btn btn-primary add-layer">Add layer</button>';
        }

        if (numNeurons[i] > 1 || (i === numNeurons.length - 1) && numNeurons[numNeurons.length - 2] === 1) {
            html += '<button type="button" class="btn btn-secondary del-neuron">Remove neuron</button>';
        }

        html += '</div>';
        html += '</div>';
    }
    $('#weights').html(html);

    function getNeuron(e) {
        var layerEl = $(e.target).closest('.layer');
        var neuronEl = $(e.target).closest('.neuron');
        var layer = $('.layer').index(layerEl);
        var neuron = layerEl.children('.neuron').index(neuronEl);
        var input = neuronEl.find('input').index(e.target);
        return [layer, neuron, input];
    }

    // Handle weight change
    $('.weight').on('input', (e) => {
        var [layer, neuron, input] = getNeuron(e);
        $(e.target).next().text(e.target.value);
        weights[layer] = weights[layer].subset(math.index(input, neuron), parseFloat(e.target.value));
        renderAR();
    });

    // Handle bias change
    $('.bias').on('input', (e) => {
        var [layer, neuron, input] = getNeuron(e);
        $(e.target).next().text(e.target.value);
        biases[layer] = biases[layer].subset(math.index(neuron), parseFloat(e.target.value));
        renderAR();
    });

    // Highlight hyperplane on hover
    $('.neuron').on('mouseenter', (e) => {
        var [layer, neuron, input] = getNeuron(e);
        highlightedEdge = layer + '/' + neuron;
        renderAR();
    });

    $('.neuron').on('mouseleave', () => {
        highlightedEdge = null;
        renderAR();
    });

    // Handle neuron add
    $('.add-neuron').on('click', (e) => {
        var [layer, neuron, input] = getNeuron(e);
        var a = layer > 0 ? numNeurons[layer - 1] : 2;
        var c = numNeurons[layer + 1];

        weights[layer] = math.concat(weights[layer], math.zeros(a, 1), 1);
        weights[layer + 1] = math.concat(weights[layer + 1], math.zeros(1, c), 0);
        biases[layer] = math.concat(biases[layer], [0]);
        numNeurons[layer]++;

        renderControls();
        renderAR();
    });

    // Handle neuron delete
    $('.del-neuron').on('click', (e) => {
        var [layer, neuron, input] = getNeuron(e);

        if (numNeurons[layer] > 1) {
            weights[layer] = dropAlongDim(weights[layer], numNeurons[layer] - 1, 1);
            weights[layer + 1] = dropAlongDim(weights[layer + 1], numNeurons[layer] - 1, 0);
            biases[layer] = dropAlongDim(biases[layer], numNeurons[layer] - 1, 0);
            numNeurons[layer]--;
        } else {
            weights.splice(layer, 1);
            biases.splice(layer, 1);
            numNeurons.splice(layer, 1);
        }

        renderControls();
        renderAR();
    });

    // Handle layer add
    $('.add-layer').on('click', (e) => {
        weights.push(math.zeros(1, 1));
        biases.push(math.zeros(1));
        numNeurons.push(1);

        renderControls();
        renderAR();
    });

    renderAR();
}

// Button controls
function initNormal() {
    initWeights(() => randn_bm() * .5, () => randn_bm() * .1);
    renderControls();
    renderAR();
}

$('#init-normal').on('click', initNormal);

$('#init-allzero').on('click', () => {
    initWeights(() => 0, () => 0);
    renderControls();
    renderAR();
});

$('#init-zerobias').on('click', () => {
    initWeights(() => randn_bm() * .1, () => 0);
    renderControls();
    renderAR();
});

/**
 * UI
 */
var numNeurons = [4, 1];
var weights = [], biases = [];
var highlightedEdge = null;


/**
 * Set up 3D scene
 */
var scene = new THREE.Scene();
scene.background = new THREE.Color( 0xffffff );

var camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight / 2, 0.1, 1000 );
camera.up = new THREE.Vector3(0, 0, 1);
camera.position.x = 0;
camera.position.y = -1;
camera.position.z = 2;
camera.lookAt(0, 0, 0);

var renderer = new THREE.WebGLRenderer({alpha: true, antialias: true});
renderer.setSize( window.innerWidth / 2, window.innerHeight );
$('body').append(renderer.domElement);

var controls = new THREE.OrbitControls(camera, renderer.domElement);

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

animate();

initNormal();
renderControls();
