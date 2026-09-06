// exposes window.DOMPurify for the CSP-restricted renderer
import DOMPurify from "dompurify";
window.DOMPurify = DOMPurify;
