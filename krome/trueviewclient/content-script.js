const YT_ORIGIN = 'https://www.youtube.com';
const YT_WATCH_PATH = 'https://www.youtube.com/watch?v=';
//const API_URL = "http://localhost:8000/result_test/";
const API_URL = "http://localhost:8000/result/";
function urlVideoParam() {
  const url = new URL(window.location.href);
  console.log("content-scirpts.js = url is = ", url);
  if (url.origin === YT_ORIGIN && url.pathname.startsWith('/watch')) {
    return url.searchParams.get('v');
  }
  return null;
}
(async () => {
  const url = API_URL + urlVideoParam();
  console.log("fetch url is ", url);
  try {
    const resp = await fetch(url, { mode:"cors" });
    if (!resp.ok) {
      throw new Error(`Response status: ${resp.status}`);
    }
    const jsonData = await resp.json();
    console.log(jsonData);
    const response = await chrome.runtime.sendMessage(jsonData);
  } catch (error) {
    console.error(error.message);
  }

})(); 

/*async function getData() {
  const url = API_URL + urlVideoParam();
  console.log("fetch url is ", url);
  try {
    const resp = await fetch(url, { mode:"cors" });
    if (!resp.ok) {
      throw new Error(`Response status: ${resp.status}`);
    }
    const jsonData = await resp.json();
    alert("jsonData = ", jsonData);
    console.log(jsonData);
    const response = await chrome.runtime.sendMessage(jsonData);
  } catch (error) {
    console.error(error.message);
  }
}
getData();
*/