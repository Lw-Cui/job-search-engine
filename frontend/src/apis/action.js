import axios from "axios";

var proxyUrl = 'https://cors-anywhere.herokuapp.com/'

export default axios.create({
  baseURL: proxyUrl + "http://backend-dot-en-601-666.ue.r.appspot.com" // /{... bm25/tfidf/bert}
});
