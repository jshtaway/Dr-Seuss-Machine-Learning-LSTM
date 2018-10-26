
// Create the Traces
var folder = "data/Alldata.json"

// var myPlot = document.getElementById('plot'),

function range(start, end) {
  var ans = [];
  for (let i = start; i <= end; i++) {
    ans.push(i);
  }
};


d3.json(folder).then(function(data) {

  var myPlot = document.getElementById('plot');
  var modelSays = data.D201810221131.sequence_list;
  
  var trace1 = {
    x: range(data.D201810221131.model_history.epochs),
    y: data.D201810221131.model_history.acc,
    mode: "markers",
    type: "scatter",
    name: "Acc",
    marker: {
      color: "#2077b4",
      symbol: "hexagram"
    }
  };

  var trace2 = {
    x: range(data.D201810221131.model_history.epochs),
    y: data.D201810221131.model_history.loss,
    mode: "markers",
    type: "scatter",
    name: "loss",
    text: modelSays,
    marker: {
      color: "orange",
      symbol: "diamond-x"
    } 
  };
  
  
  
  // Create the data array for the plot
  var data = [trace1, trace2];
  
  // Define the plot layout
  var layout = {
    title: "Dr-Suess Model",
    xaxis: { title: "Epoch" },
    yaxis: { title: "Acc/Loss" }
  };
  
  // Plot the chart to a div tag with id "plot"
   Plotly.newPlot("plot", data, layout);

  //  myPlot.on('plotly_click', function(){
  //    alert(modelSays)
  //  });







});

// var trace1 = {
//   x: range(data.D201810221131.model_history.epochs),
//   y: data.D201810221131.model_history.acc,
//   mode: "markers",
//   type: "scatter",
//   name: "high jump",
//   marker: {
//     color: "#2077b4",
//     symbol: "hexagram"
//   }
// };
// // console.log(x)
// // var trace2 = {
// //   x: data.year,
// //   y: data.discus_throw,
// //   mode: "markers",
// //   type: "scatter",
// //   name: "discus throw",
// //   marker: {
// //     color: "orange",
// //     symbol: "diamond-x"
// //   } 
// // };



// // Create the data array for the plot
// var data = [trace1];

// // Define the plot layout
// var layout = {
//   title: "Dr-Suess model",
//   xaxis: { title: "Acc" },
//   yaxis: { title: "Epochs" }
// };

// // Plot the chart to a div tag with id "plot"
// Plotly.newPlot("plot", data, layout);

