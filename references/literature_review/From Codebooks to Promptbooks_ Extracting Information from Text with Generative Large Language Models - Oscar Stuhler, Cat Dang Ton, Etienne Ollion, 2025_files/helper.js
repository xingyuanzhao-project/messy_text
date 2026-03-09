// Helper functions

// Checks if the current page is in a journal scope by examining the pbContext meta tag
function inJournalScope(){
	if (document.head.querySelector('[name="pbContext"]')==null) return false;
	return document.head.querySelector('[name="pbContext"]').content.indexOf('journal:journal:')>0;
}

// Determines if the current page is in a microsite scope
function inMicrositeScope(){
	if (document.head.querySelector('[name="pbContext"]')==null) return false;
	return document.head.querySelector('[name="pbContext"]').content.indexOf('Microsite')>0;
}

// Checks if the current page is the portal home page
function inPortalHome(){
	if (document.head.querySelector('[name="pbContext"]')==null) return false;
	return document.head.querySelector('[name="pbContext"]').content.indexOf('page:string:Home Page')>0;
}

// Determines if the current page is a search results page
function inSearchPage(){
	if (document.head.querySelector('[name="pbContext"]')==null) return false;
	return document.head.querySelector('[name="pbContext"]').content.indexOf('page:string:Search Result')>0;
}

// Checks if the current page is the publications browsing page
function inBrowsePublications(){
	if (document.head.querySelector('[name="pbContext"]')==null) return false;
	return document.head.querySelector('[name="pbContext"]').content.indexOf('page:string:Show Publications')>0;
}

// Determines if the current page is an article/chapter view page
function inArticlePage(){
	if (document.head.querySelector('[name="pbContext"]')==null) return false;
	return document.head.querySelector('[name="pbContext"]').content.indexOf('page:string:Article/Chapter View')>0;
}

// Checks if the current page is in a global scope (home, search, or browse publications)
function inGlobalScope(){
	return (inPortalHome() || inSearchPage() || inBrowsePublications());
}

// Returns the currently executing script element
function getCurrentScript(){
	if (document.currentScript)
		return document.currentScript;
}

// Wraps an HTML element with a div container having the specified ID
function wrap(org_html, tag_id){
	return new_html = '<div id="'+tag_id+'">'+ org_html.outerHTML +'</div>';
}

// Finds and returns a script element containing the specified content
function getScriptByContent(content){
	let scripts = document.getElementsByTagName("script");
	for (let i = 0; i < scripts.length; ++i)
		if(scripts[i].text.indexOf(content)!==-1)
			return scripts[i];
	return null;
}

// Returns the first element of an array if it has more than one element
function getAttribute(ar){
	if (ar.length>1)
		return ar[0];
	else return "";
}

// Returns a formatted value string from an array if it has more than one element
function getValues(ar){
	if (ar.length>1)
		return '['+ar[1];
	else return "";
}

// Removes all non-alphanumeric characters from a string
function onlyAlphanumeric(str){
	return str.replace(/[^a-z0-9]/gi, '');
}

// Generates a unique article page ID using journal code and DOI
function getArticlePageId()
{
	return 'AP'+journalAdParams.j_code+onlyAlphanumeric(journalAdParams.doi);
} 

// Generates a unique microsite page ID based on the current URL pathname
function getMicrositePageId()
{
	return 'MS'+onlyAlphanumeric(window.location.pathname).substring(0,35);
}

// Creates a page ID based on the pbContext meta tag content, cleaning and filtering specific elements
function getPbContextId()
{
	let invalidCharacters=/"|'|\^|=|\(|\)|!|<|>|\+|\[|\|#|,|\*|&|~/g;
	let pbContext=document.head.querySelector('meta[name="pbContext"]').content.replace(';','').replaceAll(invalidCharacters,'|');
	console.debug(pbContext);
	let rv='';
	pbContext.split(';').forEach(element => {
		if (element.search('ctype|csubtype|website|wgroup|pageGroup|requestedJournal|taxonomy')==-1)
			rv+=' '+element.replaceAll('string:', '').replaceAll('journal:', '').replaceAll('article:', '').replaceAll('issue:', '').replaceAll('topic:', '').replaceAll('taxonomy:', '').replaceAll('page:','').replaceAll('subPage:','');
	});
	return onlyAlphanumeric(rv.trimStart()).substring(0,39);
}

// Returns the appropriate page identifier based on the current page context
function getPageIdentifier()
{
	if (inArticlePage()) 
		return getArticlePageId();
	else if (inMicrositeScope())
		return getMicrositePageId();

	return getPbContextId();
}

// Fetches a page from the specified path using jQuery AJAX with logging
function getPage(path){
	console.debug(path);
	return $.get(path, function() {
		console.log('Getting page: ' + path+' ...');
	})
	.done(function(d) {
		console.log("Done");
	})
	.fail(function() {
		console.log("Error");
	})
}

// Uploads content to a specified path using fetch API with PUT method
const upload = async function(path, data){
	const url = location.origin+path;
	const content = data; 
	//console.debug(url);
	//console.debug(content);

	try{
		const response = await fetch(url, {
			method: 'PUT',
			headers: {
				'Content-Type': 'text/html',
			},
			body: content,
		});
	
		const text = await response.text();
		console.log(text);	
		return true;
	}
	catch(e){
		console.error(`Caught error: ${e}`);	
	}
	return false;
}

////////////////////////////////////////////////////////////
let start=Date.now(), split=Date.now();

// Resets the timer by setting both start and split times to the current time
function timerRestart(){
	start=Date.now();
	split=Date.now();
}

// Returns the elapsed time since the timer was started/restarted
function timerSplit(){
	split=Date.now();
	return split-start;
}

// Returns a formatted string showing the elapsed time since timer start
function timerSplitStr(){
	return (`[${timerSplit()}ms] `);
}

// Returns the time elapsed since the last lap and updates the split time
function timerLap(){
	let lap = Date.now()-split;
	split=Date.now();
	return lap;
}

// Returns a formatted string showing the lap time
function timerLapStr(){
	return (`[${timerLap()}ms] `);
}
/////////////////////////////////////////////////////////////

// Executes a database query with URL encoding and handles success/failure callbacks
async function runDbQuery(dbQuery, onSuccess, onFail, isTsv=false){
	let sqlParam = dbQuery.replaceAll('%','%25').replaceAll('+', '%2B').replaceAll(' ', '+').replaceAll('=', '%3D').replaceAll(',','%2C').replaceAll('#','%23').replaceAll('/','%2F');
	let tsvParam= isTsv ? "&format=tsv":"";
	//console.debug(sqlParam);
	await getPage("/action/dev?action=dbquery&q="+sqlParam+"&submit=Submit&advanced=on&pool=jdbc&size=500&lineNumbers=on&headerType=checked&headerColumnName=checked&showCompressClobs=checked&showUncompressClobs=checked"+tsvParam).always(function(dbResult) {
		try{
			//console.debug(dbResult);
			onSuccess(dbResult);
		}
		catch(e){
			onFail(e);
		}
	});
}