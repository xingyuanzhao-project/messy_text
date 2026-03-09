var dfpFile='/pb-assets/Microsite/dfp-ads'+window.location.pathname+'.json';
var dfpData=null;
var dfpParsed=false;
var dfpSlotsReady=false;

function once(fn, context) { 
	var result;
	return function() { 
		if (fn) {
			result = fn.apply(context || this, arguments);
			fn = null;
		}
		return result;
	};
}

function micrositeDfpEnabled() {
	return (dfpData!=null);
}
function getDfpInfo() {
	//console.log(dfpData);
	if (micrositeDfpEnabled()) {
		console.debug("DFP Slot = "+dfpData.dfp_slot);
		console.debug("DFP Code = "+dfpData.dfp_code);
	}
	else 
		console.debug("DFP Data = undefined");
}
function pushDfpSlots() {
	if (micrositeDfpEnabled()) {
		console.debug('Initializing DFP in Microsite scope with slot:'+dfpData.dfp_slot+' and code:'+dfpData.code);
		googletag.cmd.push(function() {
			if (document.getElementById('div-gpt-ad-' + dfpData.dfp_code + '-0')!=null){
				googletag.defineSlot('/53867575/'+dfpData.dfp_slot+'_300x250_micro', [300,250], 'div-gpt-ad-'+dfpData.dfp_code+'-0').addService(googletag.pubads());
				//console.debug('Microsite: defineSlot(/53867575/'+dfpData.dfp_slot+'_300x250_micro, [300,250], "div-gpt-ad-'+dfpData.dfp_code+'-0")');
			}
			if (document.getElementById('div-gpt-ad-' + dfpData.dfp_code + '-1')!=null){
				googletag.defineSlot('/53867575/'+dfpData.dfp_slot+'_Leader_micro_bottom', [728,90], 'div-gpt-ad-'+dfpData.dfp_code+'-1').addService(googletag.pubads());
				//console.debug('Microsite: defineSlot(/53867575/'+dfpData.dfp_slot+'_Leader_micro_bottom, [728,90], "div-gpt-ad-'+dfpData.dfp_code+'-1")');
			}
			if (document.getElementById('div-gpt-ad-' + dfpData.dfp_code + '-2')!=null){
				googletag.defineSlot('/53867575/'+dfpData.dfp_slot+'_Leader_micro_top', [728,90], 'div-gpt-ad-'+dfpData.dfp_code+'-2').addService(googletag.pubads());
				//console.debug('Microsite: defineSlot(/53867575/'+dfpData.dfp_slot+'_Leader_micro_top, [728,90], "div-gpt-ad-'+dfpData.dfp_code+'-2")');
			}
			console.log('Microsite: Pushed DFP Ads');
			//googletag.pubads().enableSingleRequest();
			googletag.pubads().setTargeting('host',document.location.host); //	Add environment targeting key-value pair
			//SAGE-6366:
			googletag.pubads().setTargeting('page',getMicrositePageId());

			googletag.pubads().collapseEmptyDivs(true);
			googletag.pubads().set("page_url", window.location.origin);
			googletag.pubads().addEventListener('slotRenderEnded', slotRendered);
			googletag.enableServices();
			dfpSlotsReady=true;
		});
	}
	else
		console.log('Microsite: CM8 Ads');
}
var pushSlotsOnce = once(pushDfpSlots);
let initPromise = function fetchJSON(input){
	console.log("initPromise");
	return new Promise((resolve ,reject)=>{
		if (dfpParsed) resolve("already parsed");
		fetch(dfpFile)
		.then(function(response) {
			if(response.ok) {
				return response.json();
			}
			dfpParsed=true;
			//console.debug(dfpFile+' NOT found');
			reject("Not found");
		})
		.then(function(data) {
			//console.debug(dfpFile+' parsed!');
			dfpData=data;
			dfpParsed=true;
			getDfpInfo();
			pushSlotsOnce();
			resolve("Parsed");
		});
	});
};
function initMicrositeDfp() {
	initPromise().then((res)=>{
		console.log(dfpFile+" "+res);
	}).catch((error)=>{
		console.error(dfpFile+" "+error);
	});
}