var plotdata = function(){
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
      autosize: false,
      width: '450',
      height: '350',
      title: "Word Based Model",
      xaxis: { title: "Epoch" },
      yaxis: { title: "Acc/Loss" }
    };
    
    // Plot the chart to a div tag with id "plot"
    Plotly.newPlot("plot", data, layout);

    //  myPlot.on('plotly_click', function(){
    //    alert(modelSays)
    //  });







});

}
plotdata()

var plotdata2 = function(){
  // Create the Traces
  var folder = "data/charjson.json"

  // var myPlot = document.getElementById('plot'),

  function range(start, end) {
    var ans = [];
    for (let i = start; i <= end; i++) {
      ans.push(i);
    }
  };


  d3.json(folder).then(function(data) {

    var myPlot = document.getElementById('plot2');
    var modelSays = data.D201810260721.sequence_list;
    
    // var trace1 = {
    //   x: range(data.D201810260721.model_history.epochs),
    //   y: data.D201810260721.model_history.acc,
    //   mode: "markers",
    //   type: "scatter",
    //   name: "Acc",
    //   marker: {
    //     color: "#2077b4",
    //     symbol: "hexagram"
    //   }
    // };

    var trace2 = {
      x: range(data.D201810260721.loss),
      y: data.D201810260721.loss,
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
    var data = [trace2];
    
    // Define the plot layout
    var layout = {
      autosize: false,
      width: '450',
      height: '350',
      title: "Character Based Model",
      xaxis: { title: "Epoch" },
      yaxis: { title: "Acc/Loss" }
    };
    
    // Plot the chart to a div tag with id "plot"
    Plotly.newPlot("plot2", data, layout);

    //  myPlot.on('plotly_click', function(){
    //    alert(modelSays)
    //  });

});

}
plotdata2()
