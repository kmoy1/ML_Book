# Test Assessment. 

<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

<div id="MCQ-question">
    <div id="q-block">
        <div id="q-text">
            Which clustering algorithms permit you to decide the number of clusters after the clustering is done?
        </div>
        <div id="q-subtext">
            <p></p>
        </div>
        <hr>
    </div>
    <div id='MCQ-block-1' style='padding: 10px;'>
        <label for='choice-1' style=' padding: 5px;'>
        <input type='radio' name='option' id='choice-1' style='transform: scale(1.6); margin-right: 10px; vertical-align: middle; margin-top: -2px;' />
        k-means clustering
        </label>
        <span id='result-1'></span>
    </div>
    <div id='MCQ-block-2' style='padding: 10px;'>
        <label for='choice-2' style=' padding: 5px;'>
        <input type='radio' name='option' id='choice-2' style='transform: scale(1.6); margin-right: 10px; vertical-align: middle; margin-top: -2px;' />
        a k-d tree used for divisive clustering
        </label>
        <span id='result-2'></span>
    </div>
    <div id='MCQ-block-3' style='padding: 10px;'>
        <label for='choice-3' style=' padding: 5px;'>
        <input type='radio' name='option' id='choice-3' style='transform: scale(1.6); margin-right: 10px; vertical-align: middle; margin-top: -2px;' />
        agglomerative clustering with single linkage
    </label>
    <span id='result-3'></span>
    </div>
    <div id='MCQ-block-4' style='padding: 10px;'>
        <label for='choice-4' style=' padding: 5px;'>
        <input type='radio' name='option' id='choice-4' style='transform: scale(1.6); margin-right: 10px; vertical-align: middle; margin-top: -2px;' />
        spectral graph clustering with 3 eigenvectors
        </label>
        <span id='result-4'></span>
    </div>
    <button type='button' onclick='checkAnswer()'>Submit</button>
</div>

<script type="text/javascript" src="http://code.jquery.com/jquery.min.js"></script>

<script>
    console.log("Started Script.");
    $.getJSON("https://my-json-server.typicode.com/kmoy1/ML_book/db", function(data){
        questions = data[0];

    });
    function checkAnswer() {
        if (document.getElementById('choice-1').checked) {
        document.getElementById('MCQ-block-1').style.border = '3px solid red'
        document.getElementById('result-1').style.color = 'red'
        document.getElementById('result-1').innerHTML = 'Incorrect!'
        }
        if (document.getElementById('choice-2').checked) {
        document.getElementById('MCQ-block-2').style.border = '3px solid limegreen'
        document.getElementById('result-2').style.color = 'limegreen'
        document.getElementById('result-2').innerHTML = 'Correct!'
        }
        if (document.getElementById('choice-3').checked) {
        document.getElementById('MCQ-block-3').style.border = '3px solid limegreen'
        document.getElementById('result-3').style.color = 'limegreen'
        document.getElementById('result-3').innerHTML = 'Correct!'
        }
        if (document.getElementById('choice-4').checked) {
        document.getElementById('MCQ-block-4').style.border = '3px solid red'
        document.getElementById('result-4').style.color = 'red'
        document.getElementById('result-4').innerHTML = 'Incorrect!'
        }
    }
</script>

<script id="MathJax-script" async
src="https://cdn.jsdelivr.net/npm/mathjax@3.0.1/es5/tex-mml-chtml.js">
</script>
</div>