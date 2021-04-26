var q_bank;
$.getJSON("https://my-json-server.typicode.com/kmoy1/ML_book/db", function(data){
    q_bank = data["questions"]
    // Populate question text + choices. 
});
q_bank.forEach(question =>{
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
                                            </div>
                                            `);
        }
        $(`div#${question.name}`).append(`<button type='button' class="btn btn-primary">Submit</button>
                                            <div class="explanation"></div>
        `);

        $(`div#${question.name} .q-text`).html(`${question.qtext}`);
        $(`div#${question.name} .MCQ-choice label`).each(function(i) {
            this.innerHTML += `${question.choices[i]}`;
        });
        submitButton = $(`div#${question.name} button`);
        submitButton.on('click', function(){
            checkAnswer2(question.name);
            if ($(`div#${question.name}`).attr("correct") === "true"){
                $(`div#${question.name} .explanation`).html(`<hr> <b>Explanation</b> <br> ${question.explanation}`);
                $(this).attr('disabled', 'disabled');  
            }
        });
    }
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
        $(`div#${questionId}`).attr("correct", "true");
    }
    else{
        $(`div#${questionId} .MCQ-choice`)[user_selected_ind].style.border = '3px solid red'
        $(`div#${questionId} .MCQ-choice span`)[user_selected_ind].style.color = 'red'
        $(`div#${questionId} .MCQ-choice span`)[user_selected_ind].innerHTML += "Incorrect.";
    }
}