$.getJSON("https://raw.githubusercontent.com/kmoy1/web_quiz/master/db.json", function(data){
    q_bank = data["questions"]
    // Populate question text + choices. 
    q_bank.forEach(question =>{
        if ($(`div#${question.name}`)){
            q_div_block = $(`div#${question.name}`);
            q_type = $(`div#${question.name}`).attr('class')
            q_div_block.html(`<div class="q-text"></div>
            <div class="q-subtext"></div>
            <hr>`);
            if (q_type === 'MCQ'){
                populateMCQ(question);
            } else if (q_type === 'Blank'){
                populateBlankQ(question);                           
            } else if (q_type === 'MCQ-All'){
                populateMCQAll(question);
            }
        }
    });
    MathJax.typeset();
});

function populateBlankQ(question){
    $(`div#${question.name}`).append(`<div class="BlankAnswer">
    <label for="inputAnswer">Your Answer</label>
    <input class="form-control" id="inputAnswer" placeholder="Password"> </div>`);
    // Append Submit Button, Correctness Logo, Explanation. 
    $(`div#${question.name}`).append(`<button type='button' class="btn btn-primary">Submit</button>
            <div class="corrlogo"></div>
            <div class="explanation"></div>`);
    $(`div#${question.name} .q-text`).html(`${question.qtext}`); 
    submitButton = $(`div#${question.name} button`);      
    submitButton.on('click', function(){ 
        user_input = $(`div#${question.name} .BlankAnswer #inputAnswer`).val();
        correct_answer = question.correctanswer;
        if (checkAnswerBlank(user_input, correct_answer)){
            $(`div#${question.name} .corrlogo`).html("Correct! &#9989;");
            $(`div#${question.name} .explanation`).html(`<hr> <b>Explanation</b> <br> ${question.explanation}`);
            $(this).attr('disabled', 'disabled');  
            $(`div#${question.name} .explanation`)
        }
        else{
            $(`div#${question.name} .corrlogo`).html("Incorrect. &#x274C;");
        }
    });
}

function populateMCQAll(question){
    q_div_block = $(`div#${question.name}`);
    num_choices = question.choices.length;
    for (let i = 0; i < num_choices; i++){
        q_div_block.append(`<div class="MCQ-All-choice">
                                            <label>
                                            <input type='checkbox' name='${question["answerlabels"][i]}'}'/>
                                            </label>
                                            <span></span>
                                        </div>
                                        `);
    }
    $(`div#${question.name}`).append(`<button type='button' class="btn btn-primary">Submit</button>
                                        <div class="corrlogo"></div>
                                        <div class="explanation"></div>
    `);

    $(`div#${question.name} .q-text`).html(`${question.qtext}`);
    $(`div#${question.name} .MCQ-All-choice label`).each(function(i) {
        this.innerHTML += `${question.choices[i]}`;
    });
    submitButton = $(`div#${question.name} button`);
    submitButton.on('click', function(){
        if (checkAnswerMCQAll(question.name)){
            $(`div#${question.name} .corrlogo`).html("Correct! &#9989;");
            $(`div#${question.name} .explanation`).html(`<hr> <b>Explanation</b> <br> ${question.explanation}`);
            $(this).attr('disabled', 'disabled');  
            $(`div#${question.name} .explanation`)
        }    
        else{
            $(`div#${question.name} .corrlogo`).html("Incorrect. &#x274C;");
        }
    });
}

function populateMCQ(question){
    q_div_block = $(`div#${question.name}`);
    num_choices = question.choices.length;
    for (let i = 0; i < num_choices; i++){
        q_div_block.append(`<div class="MCQ-choice">
                                            <label>
                                            <input type='radio' name='option'/>
                                            </label>
                                            <span></span>
                                        </div>
                                        `);
    }
    $(`div#${question.name}`).append(`<button type='button' class="btn btn-primary">Submit</button>
                                        <div class="corrlogo"></div>
                                        <div class="explanation"></div>
    `);

    $(`div#${question.name} .q-text`).html(`${question.qtext}`);
    $(`div#${question.name} .MCQ-choice label`).each(function(i) {
        this.innerHTML += `${question.choices[i]}`;
    });
    submitButton = $(`div#${question.name} button`);
    submitButton.on('click', function(){
        checkAnswerMCQ(question.name);
        if ($(`div#${question.name}`).attr("correct") === "true"){
            $(`div#${question.name} .explanation`).html(`<hr> <b>Explanation</b> <br> ${question.explanation}`);
            $(this).attr('disabled', 'disabled');  
        }
    });
}

function checkAnswerBlank(user_input, corr_answer_regex){
    var re = new RegExp(corr_answer_regex);
    return re.test(user_input);
}
function checkAnswerMCQ(questionId){
    target_q_ind = q_bank.findIndex(x => x.name === questionId);
    correct_answer = q_bank[target_q_ind].correctlabel;
    correct_answer_ind = q_bank[target_q_ind].answerlabels.indexOf(correct_answer);
    user_selected_ind = 0;
    for (let i = 0; i < q_bank[target_q_ind].choices.length; i++){
        if ($(`div#${questionId} .MCQ-choice input`)[i].checked){
            user_selected_ind = i;
        }
    }
    if (user_selected_ind == correct_answer_ind) {
        $(`div#${questionId} .corrlogo`).html("Correct! &#9989;");
        $(`div#${questionId}`).attr("correct", "true");
    }
    else{
        $(`div#${questionId} .corrlogo`).html("Incorrect. &#x274C;");
    }
}
function arrayEquals(a, b) {
    return Array.isArray(a) &&
        Array.isArray(b) &&
        a.length === b.length &&
        a.every((val, index) => val === b[index]);
}

function checkAnswerMCQAll(questionId){
    target_q_ind = q_bank.findIndex(x => x.name === questionId);
    question = q_bank[target_q_ind];
    correct_labels = question.correctlabels;
    selected = [];
    // Get selected answers into list.
    for (let i = 0; i < question.choices.length; i++){
        input_choice = $(`div#${questionId} .MCQ-All-choice input`)[i];
        if (input_choice.checked){
            selected.push(input_choice.name);
        }
    }
    console.log(selected);
    //Compare list with correct labels.
    return arrayEquals(selected, correct_labels);
}
