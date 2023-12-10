// Get the modal

var btn = document.getElementById('testbtn')
var test = null

var modal = document.getElementById("myModal");

var detailBtn = document.getElementById("detailbtn")
var details = document.getElementById("details")
var infoBtn = document.getElementById("infobtn")
var info = document.getElementById("info")
var reviewBtn = document.getElementById("reviewbtn")
var review = document.getElementById("review")
// var summaryBtn = document.getElementById("summarybtn")
// var summary = document.getElementById("summary")
var qandaBtn = document.getElementById("qandabtn")
var qanda = document.getElementById("qanda")
var answerq = document.getElementById("ans1")
// var filterBtn = document.getElementById("filterbtn")
// var filter = document.getElementById("filter")
// var filans = document.getElementById("filans")
var openBtn
var closeBtn = document.getElementById("closebtn")
// document.getElementsByClassName("modal-content").innerHTML = " <div class=\"modal-content\"> <button id=\"detailbtn\" class=\"myactionbtn\">Detail</button> <div id=\"details\" class=\"actual-content\" tabindex=\"0\" style=\"display: none;padding: 20px;\"> </div><button id=\"infobtn\" class=\"myactionbtn\">Info</button> <div id=\"info\" class=\"actual-content\" tabindex=\"0\" style=\"display: none;padding: 20px;\"> </div><button id=\"reviewbtn\" class=\"myactionbtn\">Review</button> <div id=\"review\" class=\"actual-content\" tabindex=\"0\" style=\"display: none;padding: 20px;\"> </div><button id=\"summarybtn\" class=\"myactionbtn\">Summarizer</button> <div id=\"summary\" class=\"actual-content\" tabindex=\"0\" style=\"display: none;padding: 20px;\"> </div><button id=\"qandabtn\" class=\"myactionbtn\">Q/A</button> <div id=\"qanda\" class=\"actual-content\" style=\"display: none;padding: 20px;\"> <form id=\"qandaform\"> Question: <input type=\"text\" style=\"display: inline-block;width: 80%;\" id=\"question\" name=\"qstn\"><br><!-- <input type=\"button\" id=\"submitqstn\" value=\"Submit\"> --> </form> <p tabindex=\"0\" id=\"ans1\" style=\"display: none;padding: 20px;\"> This is the answer </p></div><button id=\"filterbtn\" class=\"myactionbtn\">Filteration</button> <div id=\"filter\" class=\"actual-content\" style=\"display: none;padding: 20px;\"> <form id=\"filterform\"> Keyword: <input type=\"text\" id=\"filterword\" name=\"word\"><br><!-- <input type=\"button\" id=\"submitqstn\" value=\"Submit\"> --> </form> <p tabindex=\"0\" id=\"filans\" style=\"display: none;padding: 20px;\"> This is the answer </p></div><button class=\"close\"> <svg fill=\"currentcolor\" xmlns=\"http://www.w3.org/2000/svg\" x=\"0px\" y=\"0px\" width=\"100\" height=\"100\" viewBox=\"0 0 50 50\"> <path d=\"M 7.71875 6.28125 L 6.28125 7.71875 L 23.5625 25 L 6.28125 42.28125 L 7.71875 43.71875 L 25 26.4375 L 42.28125 43.71875 L 43.71875 42.28125 L 26.4375 25 L 43.71875 7.71875 L 42.28125 6.28125 L 25 23.5625 Z\"> </path> </svg> </button> </div>"

const hideModal = () => {
    focBtn = document.querySelectorAll('.optbtn')
    focBtn.forEach((elem, index) => {
        if (index == openBtn) {
            elem.focus()
        }
    })
    modal.style.opacity = 0;
    modal.style.pointerEvents = 'none';
    details.style.display = "none"
    info.style.display = "none"
    review.style.display = "none"
    answerq.style.display = "none"
    document.getElementById("question").value = ''
}
const showModal = () => {
    modal.style.opacity = 1;
    modal.style.pointerEvents = 'auto';
    detailBtn.focus()
}

let finaldata; // Declare a global variable

async function getJSON() {
    return fetch('./etsy.json')
        .then((response) => response.json())
        .then((responseJson) => { return responseJson });
}
async function main() {
    const json = await this.getJSON();
    console.log("json dtaa", json);
    var questions, answers, keywords
    const optionBtns = document.querySelectorAll('.optbtn')
    qandaform = document.getElementById("qandaform")
    // filterform = document.getElementById("filterform")
    optionBtns.forEach((elemtn, index) => {
        elemtn.onclick = () => {
            console.log('This button was clicked', index)
            details.innerHTML = json[index].details
            info.innerHTML = json[index].info
            review.innerHTML = json[index].review
            // summary.innerHTML = json[index].summary
            console.log('hererq');
            questions = json[index].qanda.questions
            answers = json[index].qanda.answers
            keywords = json[index].qanda.keywords
            // summaries = json[index].filter.summaries
            // sumKeys = json[index].filter.keywords
            console.log('hererq');
            openBtn = index
            showModal()
        };
    })

    qandaform.onsubmit = function (event) {

        event.preventDefault()
        var quest = (document.getElementById("question").value).toLowerCase()
        var answer = "Sorry! I do not have an answer to that question"
        for (let i = 0; i <= 5; i++) {
            if (quest.includes(keywords[i])) {
                answer = answers[i]
            }
        }
        answerq.innerHTML = answer
        answerq.style.display = "block"
        answerq.focus()
    }
    // filterform.onsubmit = function (event) {
    //     event.preventDefault()
    //     var quest = (document.getElementById("filterword").value).toLowerCase()
    //     var answer = "Sorry! No Matches Found"
    //     for (let i = 0; i <= 5; i++) {
    //         if (quest.includes(sumKeys[i])) {
    //             answer = summaries[i]
    //         }
    //     }
    //     filans.innerHTML = answer
    //     filans.style.display = "block"
    //     filans.focus()
    // }
}

main()


// var qstnSubmit = document.getElementById("submitqstn")
// btn.onclick = function () {
//     console.log("Shit is clicked")
// }

// Get the <span> element that closes the modal

var span = document.getElementsByClassName("close")[0];

// When the user clicks the button, open the modal 
btn.onclick = function () {
    // modal.style.display = "block";

    showModal();

}

// When the user clicks on <span> (x), close the modal
span.onclick = function () {
    // modal.style.display = "none";
    focBtn = document.querySelectorAll('.optbtn')
    focBtn.forEach((elem, index) => {
        if (index == openBtn) {
            elem.focus()
        }
    })
    details.style.display = "none"
    info.style.display = "none"
    review.style.display = "none"
    answerq.style.display = "none"
    qanda.style.display = "none"
    document.getElementById("question").value = ''
    modal.style.opacity = 0;


}


// When the user clicks anywhere outside of the modal, close it
window.onclick = function (event) {
    if (event.target == modal) {
        hideModal();
    }
}


document.addEventListener('keydown', (e) => {
    if (e.key === "Escape") {
        console.log('Escaaaapeddd');
        hideModal();

    }
})

closeBtn.addEventListener('keydown', (e) => {
    if (e.key === "Tab") {
        console.log('taaaab');
        hideModal();

    }
})

detailBtn.onclick = function () {
    console.log('Details Clicked', test);
    info.style.display = "none"
    review.style.display = "none"
    // summary.style.display = "none"
    qanda.style.display = "none"
    // filter.style.display = "none"
    if (test == null || test != "details") {
        details.style.display = "block"
        test = "details"

        // detailBtn.blur()
        details.focus()
    }
    else {
        details.style.display = "none"
        test = null
    }
}
infoBtn.onclick = function () {
    console.log('Information Clicked', test);
    details.style.display = "none"
    review.style.display = "none"
    // summary.style.display = "none"
    qanda.style.display = "none"
    // filter.style.display = "none"
    if (test == null || test != "info") {
        info.style.display = "block"
        test = "info"
        info.focus()
    }
    else {
        info.style.display = "none"
        test = null
    }
}
reviewBtn.onclick = function () {
    console.log('Review Clicked', test);
    details.style.display = "none"
    // summary.style.display = "none"
    qanda.style.display = "none"
    // filter.style.display = "none"
    info.style.display = "none"
    if (test == null || test != "review") {
        review.style.display = "block"
        test = "review"
        review.focus()
    }
    else {
        review.style.display = "none"
        test = null
    }
}
// summaryBtn.onclick = function () {
//     console.log('Summary Clicked', test);
//     details.style.display = "none"
//     review.style.display = "none"
//     qanda.style.display = "none"
//     filter.style.display = "none"
//     info.style.display = "none"
//     // summary.style.display = "block"
//     if (test == null || test != "summary") {
//         summary.style.display = "block"
//         test = "summary"
//         summary.focus()
//     }
//     else {
//         summary.style.display = "none"
//         test = null
//     }
// }

qandaBtn.onclick = function () {
    console.log('Q & A Clicked', test);
    // qanda.style.display = "block"
    // summary.style.display = "none"
    details.style.display = "none"
    review.style.display = "none"
    // filter.style.display = "none"
    info.style.display = "none"
    // summary.style.display = "block"
    if (test == null || test != "qanda") {
        qanda.style.display = "block"
        test = "qanda"
        document.getElementById("question").focus()
    }
    else {
        qanda.style.display = "none"
        test = null
    }
}

// qstnSubmit.onclick = function () {
//     document.getElementById("qandaform").submit()
// }

// const userQuestions = [
//     "What is the resolution of the Amazon Fire TV 4-Series Smart TV?",
//     "How does the TV's voice remote control work?",
//     "Can I stream content from Netflix and Prime Video on this TV?",
//     "Tell me about the TV's connectivity options.",
//     "What are some standout features of this smart TV according to user reviews?"
// ];

// const answers = [
//     "The TV has a 4K UHD resolution, with a display resolution of 3840 x 2160.",
//     "The TV comes with the Fire TV Alexa Voice Remote, which allows you to control functions like power, volume, and content search using voice commands.",
//     "Yes, you can stream content from popular services like Netflix and Prime Video on this smart TV.",
//     "The TV offers 3 HDMI ports, 1 USB port, and 1 Ethernet port for connecting various devices like gaming consoles and soundbars.",
//     "According to user reviews, standout features include its impressive 4K picture quality, integrated live TV streaming, Alexa voice control, easy setup, and seamless connectivity with other Amazon devices like Echo Studio and Ring cameras."
// ];



// document.getElementById("qandaform").onsubmit = function (event) {
//     event.preventDefault()
//     var quest = document.getElementById("question").value
//     var answer = null
//     console.log('form has been submitted', quest);
//     if (quest.includes("resolution")) {
//         answer = answers[0]
//     }
//     else if (quest.includes("voice")) {
//         answer = answers[1]
//     }
//     else if (quest.includes("Netflix") || quest.includes("netflix")) {
//         answer = answers[2]
//     }
//     else if (quest.includes("connectivity")) {
//         answer = answers[3]
//     }
//     else if (quest.includes("reviews")) {
//         answer = answers[4]
//     }
//     else {
//         answer = "Sorry! I do not have an answer for that question"
//     }
//     answerq.innerHTML = answer
//     answerq.style.display = "block"
//     answerq.focus()
// }

// filterBtn.onclick = function () {
//     console.log('Filter Clicked', test);
//     // filter.style.display = "block"
//     details.style.display = "none"
//     review.style.display = "none"
//     qanda.style.display = "none"
//     summary.style.display = "none"
//     info.style.display = "none"
//     // summary.style.display = "block"
//     if (test == null || test != "filter") {
//         filter.style.display = "block"
//         test = "filter"
//     }
//     else {
//         filter.style.display = "none"
//         test = null
//     }
// }

// const summaries = [
//     "The Amazon Fire TV 4-Series Smart TV boasts a stunning 4K UHD resolution with a display resolution of 3840 x 2160, providing users with an exceptional visual experience characterized by high-definition, sharp, and detailed imagery.",
//     "The TV comes equipped with the Fire TV Alexa Voice Remote, which allows users to effortlessly control various functions using voice commands, such as turning the TV on/off, adjusting volume, and searching for content, enhancing convenience and accessibility.",
//     "Users can enjoy a wide variety of content as the TV supports streaming from popular services like Netflix, Prime Video, Disney+, and more, offering a diverse selection of movies, TV shows, and other entertainment options accessible with ease.",
//     "The TV offers multiple connectivity options, including 3 HDMI ports, 1 USB port, and 1 Ethernet port, providing flexibility for connecting gaming consoles, soundbars, and other devices to create a versatile entertainment setup tailored to individual preferences.",
//     "According to user reviews, this smart TV impresses with its 4K picture quality, integrated live TV streaming, Alexa voice control functionality, user-friendly setup process, and seamless integration with other Amazon devices like Echo Studio and Ring cameras, making it a top choice for home entertainment."
// ];

// document.getElementById("filterform").onsubmit = function (event) {
//     event.preventDefault()
//     var quest = document.getElementById("filterword").value
//     var answer = null
//     console.log('filter form has been submitted', quest);
//     if (quest.includes("resolution")) {
//         answer = summaries[0]
//     }
//     else if (quest.includes("voice")) {
//         answer = summaries[1]
//     }
//     else if (quest.includes("streaming")) {
//         answer = summaries[2]
//     }
//     else if (quest.includes("connectivity")) {
//         answer = summaries[3]
//     }
//     else if (quest.includes("reviews")) {
//         answer = summaries[4]
//     }
//     else {
//         answer = "Sorry! No matches found"
//     }
//     filans.innerHTML = answer
//     filans.style.display = "block"
//     filans.focus()
// }

const elements = document.querySelectorAll('.myactionbtn');


window.onload = () => {
    console.log('loaded');
    var initv = 5
    elements.forEach((ele, index) => {
        // const elements = document.getElementById("summarybtn");
        // elements.getBoundingClientRect();
        // const width = ele.getBoundingClientRect().width

        const width = ele.clientWidth;
        console.log("width", width)
        // ele.style.left = `${10 * (index)}%`;
        ele.style.left = `${initv}%`;
        initv += width / 10 + 5
    });
}

