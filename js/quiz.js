$.getJSON("https://my-json-server.typicode.com/kmoy1/web_quiz/db", function(data){
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
            } else if (q_type === 'Blank'){
                // Append input form.
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
        }
    });
    function checkAnswerBlank(user_input, corr_answer_regex){
        var re = new RegExp(corr_answer_regex);
        return re.test(user_input);
    }
    function checkAnswerMCQ(questionId){
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
            $(`div#${questionId} .corrlogo`).html("Correct! &#9989;");
            $(`div#${questionId}`).attr("correct", "true");
        }
        else{
            $(`div#${questionId} .corrlogo`).html("Incorrect. &#x274C;");
        }
    }
});
