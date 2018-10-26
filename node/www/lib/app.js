

function loadApp() {
			
    // Create the flipbook

    $('.flipbook').turn({
            // Width

            width:922,
            
            // Height

            height:600,

            // Elevation

            elevation: 50,
            
            // Enable gradients

            gradients: true,
            
            // Auto center this flipbook

            autoCenter: true

    });
}
// function a(){
//     $.getJSON('http://localhost:5000/api/Beyonce',function(data) { alert(data);});
//     }

// $(document).keydown(function(e){

//     var previous = 37, next = 39;

//     switch (e.keyCode) {
//         case previous:

//             $('.sample-docs').turn('previous');

//         break;
//         case next:
            
//             $('.sample-docs').turn('next');

//         break;
//     }

// });

// Load the HTML4 version if there's not CSS transform

yepnope({
    test : Modernizr.csstransforms,
    yep: ['/lib/turn.js'],
    nope: ['/lib/turn.html4.min.js'],
    both: ['css/basic.css'],
    complete: loadApp
});