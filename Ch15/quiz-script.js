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
                                                    <div class="MCQ-choice">
                                                        <label>
                                                        <input type='radio' name='option'/>
                                                        </label>
                                                        <span></span>
                                                    </div>
                                                    <div class="MCQ-choice">
                                                        <label>
                                                        <input type='radio' name='option'/>
                                                        </label>
                                                        <span></span>
                                                    </div>
                                                    <div class="MCQ-choice">
                                                        <label>
                                                        <input type='radio' name='option'/>
                                                        </label>
                                                        <span></span>
                                                    </div>
                                                    <div class="MCQ-choice">
                                                        <label>
                                                        <input type='radio' name='option'/>
                                                        </label>
                                                        <span></span>
                                                    </div>
                                                    <button type='button'>Submit</button>`
                );
                $(`div#${question.name} .q-text`).html(`${question.qtext}`);
                $(`div#${question.name} .MCQ-choice label`).each(function(i) {
                    this.innerHTML += `${question.choices[i]}`;
                });
                $(`div#${question.name} button`).bind('click', function(){
                    checkAnswer2(question.name);
                })
            }
        });
    });
    function checkAnswer2(questionId){
        target_q_ind = q_bank.findIndex(x => x.name === questionId)
        correct_answer = q_bank[target_q_ind].correctlabel;
        correct_answer_ind = q_bank[target_q_ind].answerlabels.indexOf(correct_answer);
        user_selected_ind = 0
        $(`div#Question1 .MCQ-choice input`).each(i => {
            if ($(`div#${questionId} .MCQ-choice input`)[i].checked){
                user_selected_ind = i;
            }
        });

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