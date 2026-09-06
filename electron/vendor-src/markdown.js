// exposes window.markdownit for the CSP-restricted renderer
import MarkdownIt from "markdown-it";
window.markdownit = MarkdownIt;
