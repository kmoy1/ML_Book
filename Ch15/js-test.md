# Test Assessment. 

Welcome to quiz 1.

<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

<style>
input {
    transform: scale(1.6); 
    margin-right: 10px; 
    vertical-align: middle; 
    margin-top: -2px;
}
.MCQ-choice {
    padding: 10px;
}
label {
    padding: 5px;
}
</style>

<div id="Question1" class="MCQ">
</div>
<div id="Question2" class="MCQ">
</div>

<script type="text/javascript" src="http://code.jquery.com/jquery.min.js"></script>

<script>
    var q_bank;
    $.getJSON("https://my-json-server.typicode.com/kmoy1/ML_book/db", function(data){
        questions = data["questions"];
        q_bank = data["questions"]
        // Populate question text + choices. 
        questions.forEach(question =>{
            if ($(`div#${question.name}`)){
                $(`div#${question.name}`).html(`<div class="q-text">
                                                </div>
                                                <div class="q-subtext">
                                                </div>
                                                <hr>
                                                `);
                num_choices = question.choices.length;
                for (let i = 0; i < num_choices; i++){
                    $(`div#${question.name}`).append(`<div class="MCQ-choice">
                                                        <label>
                                                        <input type='radio' name='option'/>
                                                        </label>
                                                        <span></span>
                                                    </div>`);
                }
                $(`div#${question.name}`).append(`<button type='button'>Submit</button>`);

                $(`div#${question.name} .q-text`).html(`${question.qtext}`);
                $(`div#${question.name} .MCQ-choice label`).each(function(i) {
                    this.innerHTML += `${question.choices[i]}`;
                });
                $(`div#${question.name} button`).bind('click', function(){
                    checkAnswer2(question.name);
                });
            }
        });
    });
    function checkAnswer2(questionId){
        // console.log(`Checking question answer for ${questionId}`);
        target_q_ind = q_bank.findIndex(x => x.name === questionId);
        correct_answer = q_bank[target_q_ind].correctlabel;
        // console.log(`Correct Answer for this question is ${correct_answer}`);
        correct_answer_ind = q_bank[target_q_ind].answerlabels.indexOf(correct_answer);
        // console.log(`Correct Answer Index for this question is ${correct_answer_ind}`);
        user_selected_ind = 0;
        // console.log(`There are ${q_bank[target_q_ind].choices.length}$ choices for this question.`);
        for (let i = 0; i < q_bank[target_q_ind].choices.length; i++){
            if ($(`div#${questionId} .MCQ-choice input`)[i].checked){
                user_selected_ind = i;
            }
        }
        // console.log(`User Has Selected ${user_selected_ind}`);


        if (user_selected_ind == correct_answer_ind) {
            $(`div#${questionId} .MCQ-choice`)[user_selected_ind].style.border = '3px solid limegreen'
            $(`div#${questionId} .MCQ-choice span`)[user_selected_ind].style.color = 'limegreen'
            $(`div#${questionId} .MCQ-choice span`)[user_selected_ind].innerHTML += "Correct!";
        }
        else{
            $(`div#${questionId} .MCQ-choice`)[user_selected_ind].style.border = '3px solid red'
            $(`div#${questionId} .MCQ-choice span`)[user_selected_ind].style.color = 'red'
            $(`div#${questionId} .MCQ-choice span`)[user_selected_ind].innerHTML += "Incorrect.";
        }
    }
</script>

<script id="MathJax-script" async
src="https://cdn.jsdelivr.net/npm/mathjax@3.0.1/es5/tex-mml-chtml.js">
</script>
</div>