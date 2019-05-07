let model = undefined;
const TOP_K = 5

async function loadModel() {
	console.log("model loading..");
	loader = document.getElementById("progress-box")
	loader.style.display = "block";
	modelName = "mobilenet";
	model = undefined;
	model = await tf.loadLayersModel('https://frank269.github.io/MobileNet/output/mobilenet/model.json');
	loader.style.display = "none";
	console.log("model loaded..");
}

loadModel()

$(document).on('change', '#select-file-image', function() {
  	document.getElementById("select-file-box").style.display = "table-cell";
  	document.getElementById("predict-box").style.display = "table-cell";
  	document.getElementById("prediction").innerHTML = "Prediction";
    renderImage(this.files[0]);
});

function renderImage(file) {
  var reader = new FileReader();
  reader.onload = function(event) {
    img_url = event.target.result;
		document.getElementById("test-image").src = img_url;
		var image = new Image();
		image.src = img_url;
		image.onload = function(){
			predict(image);
		}
  }
  reader.readAsDataURL(file);
}
function getTopK(predictions, k){
	// Input: predictions is the output dataSync of model.predict() function
	top_k = Array.from(predictions)
		.map(function(p, i){
		    return {
		        probability: p,
		        className: IMAGENET_CLASSES[i]
		    };
		}).sort(function(a,b){
		    return b.probability - a.probability;
		}).slice(0, k);

	return top_k
}

function showResults(results){
	document.getElementById("predict-box").style.display = "block";
	document.getElementById("prediction").innerHTML = "MobileNet prediction <br><b>" + results[0].className + "</b>";

	var ul = document.getElementById("predict-list");
	ul.innerHTML = "";
	results.forEach(function (p) {
		console.log(p.className + " : " + p.probability.toFixed(6));
		var li = document.createElement("LI");
		li.innerHTML = p.className + " : " + (p.probability*100.0).toFixed(2)  + "%";
		ul.appendChild(li);
	});
}

async function predict(image) { 
	if (model == undefined) {
		alert("Please load the model first..")
	}
	console.log(model);
	//let image  = document.getElementById("test-image");
	let tensor = preprocessImage(image, modelName);
	let predictions = await model.predict(tensor).dataSync();
	showResults(getTopK(predictions, TOP_K));
}

function preprocessImage(image, modelName) {
	let tensor = tf.browser.fromPixels(image)
		.resizeNearestNeighbor([224, 224])
		.toFloat();

	if (modelName === undefined) {
		return tensor.expandDims();
	} else if (modelName === "mobilenet") {
		let offset = tf.scalar(127.5);
		return tensor.sub(offset)
			.div(offset)
			.expandDims();
	} else {
		alert("Unknown model name..")
	}
}
