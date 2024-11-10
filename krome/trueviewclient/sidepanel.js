console.log("sidepanel.js included");
function clearView(){
        //const progressbarglobal = document.getElementById("progress-bar-global");
        const analysis = document.getElementById("analysis");
        //progressbarglobal.hidden = false;
        analysis.hidden = true;
}
function makeView(data){
        //const progressbarglobal = document.getElementById("progress-bar-global");
        //progressbarglobal.hidden = true;
        const analysis = document.getElementById("analysis");
        document.getElementById("jsonresponse").textContent = JSON.stringify(data);
        document.getElementById("jsonresponse-title").textContent = data.title;
        document.getElementById("jsonresponse-summary").textContent = data.summary;
        document.getElementById("jsonresponse-viewcount").textContent = data.view_count;

        document.getElementById("jsonresponse-likes").textContent = data.no_of_likes;
        document.getElementById("inferredlikesbar").setAttribute("max", data.view_count);
        document.getElementById("inferredlikesbar").setAttribute("value", data.no_of_likes);
       
        document.getElementById("jsonresponse-dislikes").textContent = data.no_of_dislikes;
        document.getElementById("inferreddislikesbar").setAttribute("max", data.view_count);
        document.getElementById("inferreddislikesbar").setAttribute("value", data.no_of_dislikes);
        
        document.getElementById("jsonresponse-sentiment").textContent = data.sentiment_score;
        document.getElementById("sentimentbar").setAttribute("value", data.sentiment_score);
        analysis.hidden = false;
}
chrome.runtime.onMessage.addListener(
    function (request, sender, sendResponse) {
        console.log(sender.tab ?
            "recieved from a content script url:" + sender.tab.url :
            "from the extension");
       makeView(request);
       sendResponse({ msg: "sidepanel ack" });
    }
);
