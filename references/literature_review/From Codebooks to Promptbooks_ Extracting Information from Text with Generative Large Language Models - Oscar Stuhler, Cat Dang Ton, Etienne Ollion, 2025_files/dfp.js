// Google Doublickck Ad server integration:
//
// FEATURE FLAGS CONTROL:
// - DFP_FEATURE_FLAGS.CIRCUIT_BREAKER_ENABLED = false -> Disables circuit breaker complexity, uses direct execution
// - DFP_FEATURE_FLAGS.ENHANCED_LOGGING = false -> Reduces logging verbosity, disables debug/performance/context logging
//

// ===== DFP LOGGER UTILITY =====

// Standardized logging utility for DFP operations
// Provides consistent error reporting, debugging, and monitoring capabilities
class DFPLogger {
	static LOG_LEVELS = {
		ERROR: 0,
		WARN: 1,
		INFO: 2,
		DEBUG: 3
	};
	
	static currentLevel = DFPLogger.LOG_LEVELS.DEBUG; // Set based on environment
	static errorCount = 0;
	static performanceMetrics = {};
	
	// Log error with stack trace and context
	static error(message, error = null, context = {}) {
		if (this.currentLevel >= this.LOG_LEVELS.ERROR) {
			this.errorCount++;
			
			if (DFP_FEATURE_FLAGS.ENHANCED_LOGGING) {
				// Enhanced logging with full context and metadata
				const logEntry = {
					level: 'ERROR',
					timestamp: new Date().toISOString(),
					message,
					context,
					stack: error?.stack || new Error().stack,
					errorCount: this.errorCount
				};
				console.error('[DFP ERROR]', logEntry);
				this.sendToMonitoring('error', logEntry);
			} else {
				// Simple logging
				console.error('[DFP ERROR]', message, error);
			}
		}
	}
	
	// Log warning with context
	static warn(message, context = {}) {
		if (this.currentLevel >= this.LOG_LEVELS.WARN) {
			if (DFP_FEATURE_FLAGS.ENHANCED_LOGGING) {
				// Enhanced logging with full context and metadata
				const logEntry = {
					level: 'WARN',
					timestamp: new Date().toISOString(),
					message,
					context
				};
				console.warn('[DFP WARN]', logEntry);
			} else {
				// Simple logging
				console.warn('[DFP WARN]', message);
			}
		}
	}
	
	// Log informational message
	static info(message, context = {}) {
		if (this.currentLevel >= this.LOG_LEVELS.INFO) {
			if (DFP_FEATURE_FLAGS.ENHANCED_LOGGING) {
				// Enhanced logging with context
				console.info('[DFP INFO]', message, context);
			} else {
				// Simple logging
				console.info('[DFP INFO]', message);
			}
		}
	}
	
	// Log debug information
	static debug(message, context = {}) {
		if (this.currentLevel >= this.LOG_LEVELS.DEBUG) {
			if (DFP_FEATURE_FLAGS.ENHANCED_LOGGING) {
				// Enhanced logging with context
				console.debug('[DFP DEBUG]', message, context);
			} else {
				// Simple logging - skip debug in simple mode for performance
				// Only log debug messages if explicitly needed
			}
		}
	}
	
	// Track performance metrics
	static startTimer(operation) {
		if (DFP_FEATURE_FLAGS.ENHANCED_LOGGING) {
			this.performanceMetrics[operation] = {
				start: performance.now(),
				operation
			};
		}
	}
	
	static endTimer(operation, customMessage = null) {
		if (DFP_FEATURE_FLAGS.ENHANCED_LOGGING && this.performanceMetrics[operation]) {
			const duration = performance.now() - this.performanceMetrics[operation].start;
			const message = customMessage || `Performance: ${operation} completed in ${duration.toFixed(2)}ms`;
			this.info(message);
			delete this.performanceMetrics[operation];
			return duration;
		}
		return null;
	}
	
	// Circuit breaker support
	static recordFailure(service) {
		if (DFP_FEATURE_FLAGS.ENHANCED_LOGGING) {
			const key = `${service}_failures`;
			this.performanceMetrics[key] = (this.performanceMetrics[key] || 0) + 1;
			this.warn(`Service failure recorded for ${service}`, { 
				failureCount: this.performanceMetrics[key] 
			});
		}
	}
	
	static getFailureCount(service) {
		if (DFP_FEATURE_FLAGS.ENHANCED_LOGGING) {
			return this.performanceMetrics[`${service}_failures`] || 0;
		}
		return 0;
	}
	
	// Health monitoring
	static getHealthStatus() {
		if (DFP_FEATURE_FLAGS.ENHANCED_LOGGING) {
			return {
				errorCount: this.errorCount,
				activeTimers: Object.keys(this.performanceMetrics).length,
				timestamp: new Date().toISOString()
			};
		}
		return { errorCount: this.errorCount };
	}
	
	// Send to external monitoring (placeholder)
	static sendToMonitoring(level, data) {
		if (DFP_FEATURE_FLAGS.ENHANCED_LOGGING) {
			// Implement actual monitoring service integration here
			// Example: analytics service, error tracking service, etc.
		}
	}
}

// ===== CIRCUIT BREAKER PATTERN =====

// Circuit breaker implementation for external service calls
// Prevents cascading failures by temporarily disabling failing services
class CircuitBreaker {
	constructor(name, options = {}) {
		this.name = name;
		this.failureThreshold = options.failureThreshold || 5;
		this.resetTimeout = options.resetTimeout || 60000; // 1 minute
		this.monitorWindow = options.monitorWindow || 30000; // 30 seconds
		
		this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
		this.failureCount = 0;
		this.lastFailureTime = null;
		this.successCount = 0;
		this.recentFailures = [];
	}
	
	// Execute function with circuit breaker protection
	async execute(fn, fallback = null) {
		if (this.state === 'OPEN') {
			if (this.shouldAttemptReset()) {
				this.state = 'HALF_OPEN';
				DFPLogger.info(`Circuit breaker ${this.name} entering HALF_OPEN state`);
			} else {
				DFPLogger.warn(`Circuit breaker ${this.name} is OPEN, using fallback`, {
					failureCount: this.failureCount,
					lastFailure: this.lastFailureTime
				});
				return fallback ? fallback() : Promise.reject(new Error(`Circuit breaker ${this.name} is OPEN`));
			}
		}
		
		try {
			const result = await fn();
			this.onSuccess();
			return result;
		} catch (error) {
			this.onFailure();
			DFPLogger.error(`Circuit breaker ${this.name} recorded failure`, error);
			
			if (fallback) {
				DFPLogger.info(`Using fallback for ${this.name}`);
				return fallback();
			}
			throw error;
		}
	}
	
	onSuccess() {
		this.failureCount = 0;
		this.successCount++;
		
		if (this.state === 'HALF_OPEN') {
			this.state = 'CLOSED';
			DFPLogger.info(`Circuit breaker ${this.name} reset to CLOSED state`);
		}
	}
	
	onFailure() {
		this.failureCount++;
		this.lastFailureTime = Date.now();
		this.recentFailures.push(this.lastFailureTime);
		
		// Clean old failures outside monitor window
		this.recentFailures = this.recentFailures.filter(
			time => Date.now() - time < this.monitorWindow
		);
		
		if (this.recentFailures.length >= this.failureThreshold) {
			this.state = 'OPEN';
			DFPLogger.error(`Circuit breaker ${this.name} opened due to repeated failures`, null, {
				failureCount: this.failureCount,
				recentFailures: this.recentFailures.length
			});
		}
	}
	
	shouldAttemptReset() {
		return Date.now() - this.lastFailureTime > this.resetTimeout;
	}
	
	getState() {
		return {
			name: this.name,
			state: this.state,
			failureCount: this.failureCount,
			successCount: this.successCount,
			lastFailureTime: this.lastFailureTime,
			recentFailures: this.recentFailures.length
		};
	}
}

// Service-specific circuit breakers
const aimTagCircuitBreaker = new CircuitBreaker('AimTag', {
	failureThreshold: 3,
	resetTimeout: 120000, // 2 minutes
	monitorWindow: 60000 // 1 minute
});

const googleTagCircuitBreaker = new CircuitBreaker('GoogleTag', {
	failureThreshold: 5,
	resetTimeout: 60000, // 1 minute
	monitorWindow: 30000 // 30 seconds
});

// ===== CIRCUIT BREAKER UTILITY FUNCTIONS =====

// Execute a function with optional circuit breaker protection based on feature flag
// When circuit breaker is disabled, executes the function directly with basic error handling
async function executeWithCircuitBreaker(circuitBreaker, operation, fallback = null, context = '') {
	if (DFP_FEATURE_FLAGS.CIRCUIT_BREAKER_ENABLED) {
		// Use circuit breaker pattern
		return circuitBreaker.execute(operation, fallback);
	} else {
		// Direct execution with basic error handling
		try {
			DFPLogger.debug(`Executing ${context} without circuit breaker`);
			return await operation();
		} catch (error) {
			DFPLogger.error(`${context} failed, using fallback`, error);
			if (fallback) {
				return fallback();
			}
			throw error;
		}
	}
}

// ===== CONFIGURATION OBJECTS =====

// Feature flags for controlling advanced functionality
const DFP_FEATURE_FLAGS = {
	CIRCUIT_BREAKER_ENABLED: false, // Set to false to disable circuit breaker pattern
	ENHANCED_LOGGING: false          // Set to false to use basic console logging only
};

// Main DFP configuration constants
const DFP_CONFIG = {
	NETWORK_ID: '53867575',
	JOURNAL_CODE: '123456789',
	RETRY_INTERVAL: 500,
	DEFAULT_MAX_WIDTH: 9999,
	DEFAULT_MIN_WIDTH: 0,
	SLOT_ELEMENT_IDS: {
		TOP_SKYSCRAPER: 'top-skyscraper-placeholder',
		BOTTOM_SKYSCRAPER: 'bottom-skyscraper-placeholder',
		AD_KEYWORDS: 'ad-keywords'
	}
};

// Ad slot size definitions
const AD_SLOT_SIZES = {
	LEADERBOARD: [728, 90],
	MEDIUM_RECTANGLE: [300, 250],
	SKYSCRAPER: [160, 600],
	WIDE_SKYSCRAPER: [300, 600],
	MULTI_SIZE_SKYSCRAPER: [[160, 600], [300, 250], [300, 600]]
};

// Ad slot placeholder configurations
const AD_SLOT_CONFIG = {
	NORMAL_PLACEHOLDERS: ['728x90_Top', '300x250_Top', '300x250_bottom', '728x90_bottom', 'Interstitial', ''],
	MAL_PLACEHOLDERS: ['728x90_Leaderboard', '300x250_Box1', '300x250_Box2', '', 'Interstitial', '728x90_Footer'],
	JOURNAL_SLOTS: {
		0: { size: AD_SLOT_SIZES.LEADERBOARD, type: 'top' },
		1: { size: AD_SLOT_SIZES.MEDIUM_RECTANGLE, type: 'sidebar' },
		2: { size: AD_SLOT_SIZES.MULTI_SIZE_SKYSCRAPER, type: 'skyscraper' },
		3: { size: AD_SLOT_SIZES.LEADERBOARD, type: 'bottom' },
		4: { type: 'interstitial' },
		5: { size: AD_SLOT_SIZES.LEADERBOARD, type: 'footer' }
	}
};

// AimTag configuration
const AIMTAG_CONFIG = {
	API_KEY: 'f5aacc07-e894-49ce-9b98-2f2bcd40b55b',
	TARGETING_KEY: 'aim_signal_specialty'
};

// Doceree ad configuration
const DOCEREE_CONFIG = {
	ENABLED_INDICES: [0, 2],
	TAG_PREFIX: 'div-doceree-ad'
};

// Publisher and society configuration
const PUBLISHER_CONFIG = {
	MAL_SOCIETY_NAME: 'mary ann liebert',
	SERVICE_NAME_DFP: 'DFP'
};

// ===== END CONFIGURATION =====

// ===== STRATEGY PATTERN FOR PAGE TYPES =====

// Base strategy class for different page types
// Defines the interface that all page type strategies must implement
class PageTypeStrategy {
	constructor(name) {
		this.name = name;
	}
	
	// Generate ad tag ID for this page type
	generateTagId(index) {
		throw new Error('generateTagId must be implemented by subclass');
	}
	
	// Initialize DFP for this page type
	initializeDfp() {
		throw new Error('initializeDfp must be implemented by subclass');
	}
	
	// Check if this strategy applies to current page context
	isApplicable() {
		throw new Error('isApplicable must be implemented by subclass');
	}
	
	// Get debug message for this page type
	getDebugMessage(index) {
		return `${index}: ${this.name} DFP`;
	}
}

// Journal page strategy
class JournalPageStrategy extends PageTypeStrategy {
	constructor() {
		super('Journal');
	}
	
	isApplicable() {
		return inJournalScope();
	}
	
	generateTagId(index) {
		// Validate that the requested slot index is valid for current journal type
		if (index === 3 && isMalJournal()) {
			console.warn(`Ad slot index ${index} is not available for MAL journals`);
			return null;
		}
		
		if (index === 5 && !isMalJournal()) {
			console.warn(`Ad slot index ${index} is not available for non-MAL journals`);
			return null;
		}
		
		if (docereeEnabled && DOCEREE_CONFIG.ENABLED_INDICES.includes(index)) {
			console.debug(`${index}: Doceree Ads`);
			return DOCEREE_CONFIG.TAG_PREFIX + index.toString();
		} else {
			console.debug(this.getDebugMessage(index));
			return 'div-gpt-ad-' + DFP_CONFIG.JOURNAL_CODE + '-' + index.toString();
		}
	}
	
	initializeDfp() {
		initJournalDfp();
	}
}

// Microsite page strategy
class MicrositePageStrategy extends PageTypeStrategy {
	constructor() {
		super('Microsite');
	}
	
	isApplicable() {
		return inMicrositeScope();
	}
	
	generateTagId(index) {
		console.debug(this.getDebugMessage(index));
		return 'div-gpt-ad-' + dfpData.dfp_code + '-' + index.toString();
	}
	
	initializeDfp() {
		// Microsite initialization handled elsewhere in the codebase
		console.debug('Microsite DFP initialization');
	}
}

// Portal home page strategy
class PortalPageStrategy extends PageTypeStrategy {
	constructor() {
		super('Portal');
	}
	
	isApplicable() {
		return inPortalHome();
	}
	
	generateTagId(index) {
		console.debug(this.getDebugMessage(index));
		return 'div-gpt-ad-' + globalAdParams.portal.code + '-' + index.toString();
	}
	
	initializeDfp() {
		initPageDfp(globalAdParams.portal);
	}
}

// Search page strategy
class SearchPageStrategy extends PageTypeStrategy {
	constructor() {
		super('Search');
	}
	
	isApplicable() {
		return inSearchPage();
	}
	
	generateTagId(index) {
		console.debug(this.getDebugMessage(index));
		return 'div-gpt-ad-' + globalAdParams.search.code + '-' + index.toString();
	}
	
	initializeDfp() {
		initPageDfp(globalAdParams.search);
	}
}

// Browse publications page strategy
class BrowsePageStrategy extends PageTypeStrategy {
	constructor() {
		super('Browse');
	}
	
	isApplicable() {
		return inBrowsePublications();
	}
	
	generateTagId(index) {
		console.debug(this.getDebugMessage(index));
		return 'div-gpt-ad-' + globalAdParams.browse.code + '-' + index.toString();
	}
	
	initializeDfp() {
		initPageDfp(globalAdParams.browse);
	}
}

// Page type context that manages strategy selection and execution
class PageTypeContext {
	constructor() {
		this.strategies = [
			new JournalPageStrategy(),
			new MicrositePageStrategy(),
			new PortalPageStrategy(),
			new SearchPageStrategy(),
			new BrowsePageStrategy()
		];
	}
	
	// Get the appropriate strategy for current page context
	getCurrentStrategy() {
		return this.strategies.find(strategy => strategy.isApplicable()) || null;
	}
	
	// Generate tag ID using current page strategy
	generateTagId(index) {
		const strategy = this.getCurrentStrategy();
		if (!strategy) {
			console.warn('No valid page context found for ad index: ' + index);
			return null;
		}
		return strategy.generateTagId(index);
	}
	
	// Initialize DFP using current page strategy
	initializeDfp() {
		const strategy = this.getCurrentStrategy();
		if (!strategy) {
			console.warn('DFP ads will not be rendered: No proper page scope is detected!');
			return false;
		}
		strategy.initializeDfp();
		return true;
	}
}

// Global instance of page type context
const pageTypeContext = new PageTypeContext();

// ===== END STRATEGY PATTERN =====

// Checks if the current journal belongs to Mary Ann Liebert publishers
function isMalJournal(){
	return (journalAdParams.society.toLowerCase().indexOf(PUBLISHER_CONFIG.MAL_SOCIETY_NAME) != -1);
}

// Determines if Google DFP (DoubleClick for Publishers) ads are enabled for the current page context
// Checks different scopes (journal, microsite, global) and their respective DFP configuration
function dfpEnabled(){
	let rv=false;
	//return rv; // Remove (or comment) this line to enable google DFP ads (SAGE-2584)
	if (inJournalScope())
		rv=(journalAdParams.serviceName.indexOf(PUBLISHER_CONFIG.SERVICE_NAME_DFP)>=0 || isMalJournal());
	else if (inMicrositeScope())
		rv=micrositeDfpEnabled();
	else if (inGlobalScope())
		rv=(globalAdParams!==null);

	if (!rv) DFPLogger.warn("DFP is not enabled for current context", {
		inJournal: inJournalScope(),
		inMicrosite: inMicrositeScope(),
		inGlobal: inGlobalScope()
	});
	return rv;
}

// Moves skyscraper ad content from bottom placeholder to top placeholder if visible
// Used for repositioning tall banner ads based on page layout
function moveSkyscraperAd(){
	try {
		const topAd = document.getElementById(DFP_CONFIG.SLOT_ELEMENT_IDS.TOP_SKYSCRAPER);
		const bottomAd = document.getElementById(DFP_CONFIG.SLOT_ELEMENT_IDS.BOTTOM_SKYSCRAPER);
		
		if (!topAd) {
			DFPLogger.debug('Top skyscraper placeholder element not found', { 
				elementId: DFP_CONFIG.SLOT_ELEMENT_IDS.TOP_SKYSCRAPER 
			});
			return false;
		}
		
		if (!bottomAd) {
			DFPLogger.debug('Bottom skyscraper placeholder element not found', { 
				elementId: DFP_CONFIG.SLOT_ELEMENT_IDS.BOTTOM_SKYSCRAPER 
			});
			return false;
		}
		
		// Check if top ad is visible before moving content
		if (!topAd.checkVisibility || !topAd.checkVisibility()) {
			DFPLogger.debug('Top skyscraper placeholder not visible, skipping move');
			return false;
		}
		
		// Safely move child nodes
		const childNodes = Array.from(bottomAd.childNodes);
		childNodes.forEach((item) => {
			try {
				topAd.appendChild(item);
			} catch (moveError) {
				DFPLogger.error('Failed to move child node', moveError, { 
					nodeType: item.nodeType,
					nodeName: item.nodeName 
				});
			}
		});
		
		DFPLogger.info('Skyscraper ad moved up successfully', { 
			movedNodes: childNodes.length 
		});
		return true;
		
	} catch (error) {
		DFPLogger.error('Failed to move skyscraper ad', error);
		return false;
	}
}

// Sets up event listener for PubGrade contextual targeting completion
// Displays the ad after PubGrade analysis is finished with timeout protection
function pubGradePush(tagId){
	DFPLogger.debug('Setting up PubGrade event listener with timeout protection', { tagId });
	
	let eventHandled = false;
	const timeout = 15000; // 15 second timeout
	
	// PubGrade completion handler
	const pubGradeHandler = function() {
		if (eventHandled) return;
		eventHandled = true;
		
		const googleTagOperation = () => {
			return new Promise((resolve, reject) => {
				try {
					googletag.cmd.push(function() { googletag.display(tagId); });
					DFPLogger.debug('PubGrade completed - ad display triggered', { tagId });
					resolve();
				} catch (error) {
					reject(error);
				}
			});
		};
		
		const googleTagFallback = () => {
			DFPLogger.warn('PubGrade Google Tag display failed, skipping ad', { tagId });
			return Promise.resolve();
		};
		
		executeWithCircuitBreaker(googleTagCircuitBreaker, googleTagOperation, googleTagFallback, 'PubGrade Google Tag')
			.catch((error) => {
				DFPLogger.error('PubGrade ad display failed', error, { tagId });
			});
	};
	
	// Timeout handler
	const timeoutHandler = setTimeout(() => {
		if (eventHandled) return;
		eventHandled = true;
		
		DFPLogger.warn('PubGrade timeout - proceeding without contextual targeting', { 
			tagId, 
			timeoutMs: timeout 
		});
		
		// Fallback to regular ad display
		pushWhenReady(tagId);
	}, timeout);
	
	// Set up event listener
	document.addEventListener('pbgrdFinished', function() {
		clearTimeout(timeoutHandler);
		pubGradeHandler();
	}, { once: true });
}

// Waits for DFP slots to be ready before displaying ads
// Retries every 500ms if slots are not yet initialized with timeout protection
function pushWhenReady(tagId, maxRetries = 20, currentRetry = 0){
	if(dfpSlotsReady) {
		const googleTagOperation = () => {
			return new Promise((resolve, reject) => {
				try {
					googletag.cmd.push(function() { 
						googletag.display(tagId);
						resolve();
					});
					DFPLogger.debug('Google Tag display command pushed successfully', { tagId });
				} catch (error) {
					reject(error);
				}
			});
		};
		
		const googleTagFallback = () => {
			DFPLogger.warn('Google Tag display failed, skipping ad', { tagId });
			return Promise.resolve();
		};
		
		// Execute with circuit breaker protection
		executeWithCircuitBreaker(googleTagCircuitBreaker, googleTagOperation, googleTagFallback, 'Google Tag Display')
			.catch((error) => {
				DFPLogger.error('Failed to display Google Tag', error, { tagId });
			});
	}
	else {
		if (currentRetry >= maxRetries) {
			DFPLogger.error('Timeout waiting for DFP slots to be ready', null, { 
				tagId, 
				maxRetries,
				timeoutMs: maxRetries * DFP_CONFIG.RETRY_INTERVAL
			});
			return;
		}
		
		DFPLogger.debug('DFP slots not ready, retrying...', { 
			tagId, 
			retry: currentRetry + 1, 
			maxRetries 
		});
		setTimeout(pushWhenReady, DFP_CONFIG.RETRY_INTERVAL, tagId, maxRetries, currentRetry + 1);
	}
}

// Event handler for when Google Ad slots finish rendering
// Logs ad details and performs post-render actions like moving related content
function slotRendered(event) {
	var slot = event.slot;
	var domElementId = slot.getSlotElementId();
	DFPLogger.info('Ad slot finished rendering: '+ domElementId, {
		slotId: domElementId,
		advertiserId: event.advertiserId,
		campaignId: event.campaignId,
		creativeId: event.creativeId,
		isEmpty: event.isEmpty,
		lineItemId: event.lineItemId,
		size: event.size
	});

	//LIT-660656 -  Close button in MAL footer ad
	if(event.slot.getSlotElementId() == generateAdTagId(5) && !event.isEmpty) {
		$("#"+generateAdTagId(5)).append('<style> .close-ad-btn{ max-width: 25px; max-height: 25px; width: auto; height: auto; margin: 0px; background: transparent 0px center; border: 0px; box-shadow: none; content: normal; direction: ltr; float: none; line-height: normal; outline: none; display: block; position: absolute; bottom: 60px; cursor: pointer; padding: 10px; box-sizing: content-box; z-index: 7000002; transform: scale(1); right: -17px}</style>');
		$("#"+generateAdTagId(5)).append('<input type="image" alt="Close" aria-label="Close ad" src="/pb-assets/ux3/images/interstitial-close.svg" id="mal-footer-ad-close" class="close-ad-btn"></input>');
		
		$(".close-ad-btn").click(function () {
			$(this).parent().fadeOut(800);
		});
	}	
	
	if (domElementId==='div-gpt-ad-123456789-2') {
		DFPLogger.debug('Skyscraper ad rendered - related actions could be triggered here');
		//moveRelatedJournals();
	}
}

// Generates the appropriate ad tag ID based on page context and ad index
// Uses strategy pattern to handle different page types
function generateAdTagId(index) {
	return pageTypeContext.generateTagId(index);
}

// Creates the ad container HTML and inserts it into the DOM
// Returns true if successful, false if script element not found
function createAdContainer(tagId, index) {
	try {
		let script = getCurrentScript();
		if (!script) {
			script = getScriptByContent('showDfpAd(' + index);
		}
		
		if (!script) {
			DFPLogger.error('Script element not found for ad container creation', null, { 
				tagId, 
				index,
				searchPattern: 'showDfpAd(' + index
			});
			return false;
		}
		
		const parent = script.parentNode;
		if (!parent) {
			DFPLogger.error('Script element has no parent node', null, { 
				tagId, 
				index 
			});
			return false;
		}
		
		// Safely create the ad container HTML
		const containerHtml = wrap(script, tagId);
		if (!containerHtml) {
			DFPLogger.error('Failed to generate container HTML', null, { 
				tagId, 
				index 
			});
			return false;
		}
		
		parent.innerHTML = containerHtml;
		DFPLogger.debug('Ad container created successfully', { 
			tagId, 
			index 
		});
		return true;
		
	} catch (error) {
		DFPLogger.error('Exception in createAdContainer', error, { 
			tagId, 
			index 
		});
		return false;
	}
}

// Handles the display logic for different ad types
// Manages doceree ads, article page ads, and regular ad display
function displayAdSlot(tagId, index) {
	if (tagId.indexOf('doceree') !== -1) {
		console.error('Doceree ads are disabled');
		// $('#' + tagId).load("/pb-assets/ads/ads-" + journalAdParams.alt_code + "-" + index + "-doceree.html");
	} else {
		if (inArticlePage()) {
			moveSkyscraperAd();
			pubGradePush(tagId); // SAGE-6133: PubGrade contextual targeting
		} else {
			console.debug("Pushing when ready: tagId=" + tagId);
			pushWhenReady(tagId);
		}
	}
}

// Renders an ad based on the index and current page context (journal, microsite, portal, etc.)
// Creates the appropriate HTML container and triggers ad display
function renderAd(index){
	if (!dfpEnabled()) {
		return false;
	}

	const tagId = generateAdTagId(index);
	if (!tagId) {
		console.warn('No valid page context found for ad index: ' + index);
		return false;
	}

	if (!createAdContainer(tagId, index)) {
		return false;
	}

	displayAdSlot(tagId, index);
	return true;
}

// Main entry point function called from page inline scripts to display DFP ads
// Checks window width constraints and delegates to renderAd for actual rendering
function showDfpAd(index, minWidth=DFP_CONFIG.DEFAULT_MIN_WIDTH, maxWidth=DFP_CONFIG.DEFAULT_MAX_WIDTH){
	try {
		// Window width validation
		if (window.innerWidth < minWidth || window.innerWidth > maxWidth){
			DFPLogger.debug('Ad slot not shown due to window width constraints', {
				index,
				windowWidth: window.innerWidth,
				minWidth,
				maxWidth
			});
			return Promise.resolve(false);
		}
		
		DFPLogger.debug('Attempting to show DFP ad', { index });
		
		if (!inMicrositeScope()) {
			const result = renderAd(index);
			return Promise.resolve(result);
		} else {
			// Enhanced microsite promise handling with timeout
			const initPromiseWithTimeout = Promise.race([
				initPromise(),
				new Promise((_, reject) => 
					setTimeout(() => reject(new Error('Microsite initialization timeout')), 10000)
				)
			]);
			
			return initPromiseWithTimeout
				.then((res) => {
					DFPLogger.debug('Microsite initialization completed', { index, result: res });
					return renderAd(index);
				})
				.catch((error) => {
					DFPLogger.error('Microsite initialization failed', error, { 
						index,
						file: dfpFile || 'dfp.js'
					});
					
					// Attempt graceful degradation - try to render ad anyway
					DFPLogger.info('Attempting graceful degradation for microsite ad', { index });
					try {
						return renderAd(index);
					} catch (fallbackError) {
						DFPLogger.error('Graceful degradation also failed', fallbackError, { index });
						return false;
					}
				});
		}
	} catch (error) {
		DFPLogger.error('Unexpected error in showDfpAd', error, { 
			index, 
			minWidth, 
			maxWidth 
		});
		return Promise.resolve(false);
	}
}

// Generates the Google Ad Manager slot name based on journal type and ad index
// Uses different slot naming conventions for Mary Ann Liebert journals vs. normal journals
function getSlotName(index){
	if (isMalJournal()){
		return '/' + DFP_CONFIG.NETWORK_ID + '/' + journalAdParams.alpha_code + '/' + AD_SLOT_CONFIG.MAL_PLACEHOLDERS[index];
	}
	else{
		return '/' + DFP_CONFIG.NETWORK_ID + '/' + journalAdParams.dfp_slot + '//' + journalAdParams.alpha_code + '//' + AD_SLOT_CONFIG.NORMAL_PLACEHOLDERS[index];
	}
}

// Defines all Google Ad slots for journal pages based on journal type
// Handles different slot configurations for Mary Ann Liebert vs. normal journals
function defineJournalAdSlots() {
	// Slot 0: Top Leaderboard (728x90)
	googletag.defineSlot(getSlotName(0), AD_SLOT_SIZES.LEADERBOARD, 'div-gpt-ad-' + DFP_CONFIG.JOURNAL_CODE + '-0').addService(googletag.pubads());
	
	// Slot 1: Medium Rectangle (300x250)
	googletag.defineSlot(getSlotName(1), AD_SLOT_SIZES.MEDIUM_RECTANGLE, 'div-gpt-ad-' + DFP_CONFIG.JOURNAL_CODE + '-1').addService(googletag.pubads());
	
	// Slot 2: Skyscraper (Multiple sizes)
	googletag.defineSlot(getSlotName(2), AD_SLOT_SIZES.MULTI_SIZE_SKYSCRAPER, 'div-gpt-ad-' + DFP_CONFIG.JOURNAL_CODE + '-2').addService(googletag.pubads());
	
	// Slot 3: Bottom Leaderboard (728x90) - Only for non-MAL journals
	if (!isMalJournal()) {
		if (document.getElementById('div-gpt-ad-' + DFP_CONFIG.JOURNAL_CODE + '-3')!=null){
			googletag.defineSlot(getSlotName(3), AD_SLOT_SIZES.LEADERBOARD, 'div-gpt-ad-' + DFP_CONFIG.JOURNAL_CODE + '-3').addService(googletag.pubads());
		}
	}
	
	// Slot 4: Interstitial (out-of-page)
	if (document.getElementById('div-gpt-ad-' + DFP_CONFIG.JOURNAL_CODE + '-4')!=null){
		googletag.defineOutOfPageSlot(getSlotName(4), 'div-gpt-ad-' + DFP_CONFIG.JOURNAL_CODE + '-4').addService(googletag.pubads());
	}

	// Slot 5: Footer Leaderboard (728x90) - Only for MAL journals
	if (isMalJournal()) {
		googletag.defineSlot(getSlotName(5), AD_SLOT_SIZES.LEADERBOARD, 'div-gpt-ad-' + DFP_CONFIG.JOURNAL_CODE + '-5').addService(googletag.pubads());
	}
}

// Sets up AimTag integration for medical audience targeting
// Handles both signal detection and async rendering enablement
function setupAimTagIntegration() {
	const aimTagOperation = () => {
		return new Promise((resolve, reject) => {
			aimTag(AIMTAG_CONFIG.API_KEY, 'signal', function(err, success) {
				if (err) {
					reject(new Error('AimTag error: ' + err));
				} else {
					googletag.pubads().setTargeting(AIMTAG_CONFIG.TARGETING_KEY, [success.primary_specialty_code]);
					DFPLogger.debug('AimTag signal received successfully', { 
						specialty: success.primary_specialty_code 
					});
					googletag.pubads().enableAsyncRendering();
					DFPLogger.debug('AimTag async rendering enabled');
					resolve(success);
				}
			});
		});
	};
	
	const aimTagFallback = () => {
		DFPLogger.warn('Using AimTag fallback - proceeding without medical targeting');
		googletag.pubads().enableAsyncRendering();
		return Promise.resolve({ fallback: true });
	};
	
	// Execute AimTag with circuit breaker protection
	executeWithCircuitBreaker(aimTagCircuitBreaker, aimTagOperation, aimTagFallback, 'AimTag Integration')
		.then((result) => {
			if (!result.fallback) {
				DFPLogger.info('AimTag integration completed successfully');
			}
		})
		.catch((error) => {
			DFPLogger.error('AimTag integration failed completely', error);
			// Ensure async rendering is enabled even if AimTag fails
			googletag.pubads().enableAsyncRendering();
		});
}

// Sets up all DFP targeting parameters for journal pages
// Includes host, page identifier, custom targeting, and taxonomy keywords
function setupDfpTargeting() {
	// Add environment targeting key-value pair
	googletag.pubads().setTargeting('host', document.location.host);
	
	// SAGE-6366: Page identifier targeting
	googletag.pubads().setTargeting('page', getPageIdentifier());
	
	// SAGE-6040: Custom ad targeting parameters
	if (journalAdParams.ad_targeting.length > 1) {
		googletag.pubads().setTargeting(getAttribute(journalAdParams.ad_targeting), getValues(journalAdParams.ad_targeting));
	}
	
	// SAGE-7220 + SAGE-7662: Taxonomy keywords targeting
	try {
		const keywordsElement = document.getElementById(DFP_CONFIG.SLOT_ELEMENT_IDS.AD_KEYWORDS);
		if (!keywordsElement) {
			DFPLogger.debug('Ad keywords element not found - skipping taxonomy targeting', {
				elementId: DFP_CONFIG.SLOT_ELEMENT_IDS.AD_KEYWORDS
			});
			return;
		}
		
		const textContent = keywordsElement.textContent;
		if (!textContent || typeof textContent !== 'string') {
			DFPLogger.debug('Ad keywords element has no text content - skipping taxonomy targeting');
			return;
		}
		
		const keywords = textContent.trimEnd();
		if (keywords.length > 0) {
			const keywordArray = keywords.substring(0, keywords.length - 1)
				.replace(/\s+/g, '')
				.split(',')
				.filter(keyword => keyword.length > 0); // Remove empty keywords
			
			if (keywordArray.length > 0) {
				googletag.pubads().setTargeting('taxonomy', keywordArray);
				DFPLogger.debug('Taxonomy targeting set successfully', { 
					keywordCount: keywordArray.length,
					keywords: keywordArray 
				});
			} else {
				DFPLogger.debug('No valid keywords found after processing');
			}
		} else {
			DFPLogger.debug('Keywords element is empty - skipping taxonomy targeting');
		}
	} catch(error) {
		DFPLogger.error('Failed to set taxonomy targeting', error, {
			elementId: DFP_CONFIG.SLOT_ELEMENT_IDS.AD_KEYWORDS
		});
	}
}

// Enables and configures Google DFP services
// Sets up event listeners, page URL, and collapse empty divs
function enableDfpServices() {
	googletag.pubads().collapseEmptyDivs(true);
	googletag.pubads().set("page_url", window.location.origin);
	googletag.pubads().addEventListener('slotRenderEnded', slotRendered);
	googletag.enableServices();
	dfpSlotsReady = true;
}

// Initializes Google DFP ad slots specifically for journal pages
// Sets up targeting parameters, AimTag integration, and ad slot definitions
function initJournalDfp(){
	DFPLogger.info('Initializing DFP for journal scope', {
		slot: journalAdParams.dfp_slot,
		code: DFP_CONFIG.JOURNAL_CODE,
		isMal: isMalJournal()
	});
	
	if (dfpEnabled()) {
		DFPLogger.startTimer('journal_dfp_init');
		
		googletag.cmd.push(function(){
			try {
				defineJournalAdSlots();
				setupAimTagIntegration();
				
				DFPLogger.info('Journal DFP slots defined successfully');
				
				setupDfpTargeting();
				enableDfpServices();
				
				DFPLogger.endTimer('journal_dfp_init');
			} catch (error) {
				DFPLogger.error('Failed to initialize journal DFP', error);
			}
		});
	} else {
		DFPLogger.info('Journal: Using CM8 Ads (DFP disabled)');
	}
}

// Initializes Google DFP ad slots for non-journal pages (portal, search, browse)
// Creates standard ad slots with page-specific targeting and configuration
function initPageDfp(page){
	DFPLogger.info('Initializing DFP for page scope', {
		name: page.name,
		slot: page.slot,
		code: page.code
	});
	
	if (dfpEnabled()){
		DFPLogger.startTimer('page_dfp_init');
		
		googletag.cmd.push(function(){
			try {
				// Define all page ad slots
				googletag.defineSlot('/' + DFP_CONFIG.NETWORK_ID + '/' + page.slot + '//728x90_Top', AD_SLOT_SIZES.LEADERBOARD, 'div-gpt-ad-' + page.code + '-0').addService(googletag.pubads());
				googletag.defineSlot('/' + DFP_CONFIG.NETWORK_ID + '/' + page.slot + '//300x250_Top', AD_SLOT_SIZES.MEDIUM_RECTANGLE, 'div-gpt-ad-' + page.code + '-1').addService(googletag.pubads());
				googletag.defineSlot('/' + DFP_CONFIG.NETWORK_ID + '/' + page.slot + '//300x250_bottom', AD_SLOT_SIZES.MULTI_SIZE_SKYSCRAPER, 'div-gpt-ad-' + page.code + '-2').addService(googletag.pubads());
				if (document.getElementById('div-gpt-ad-' + DFP_CONFIG.JOURNAL_CODE + '-3')!=null){
					googletag.defineSlot('/' + DFP_CONFIG.NETWORK_ID + '/' + page.slot + '//728x90_bottom', AD_SLOT_SIZES.LEADERBOARD, 'div-gpt-ad-' + page.code + '-3').addService(googletag.pubads());
				}

				DFPLogger.info(`${page.name}: DFP slots defined`);
				
				googletag.pubads().setTargeting('host',document.location.host); //	Add environment targeting key-value pair
				//SAGE-6366:
				googletag.pubads().setTargeting('page',getPageIdentifier());
				//googletag.pubads().enableSingleRequest();
				googletag.pubads().collapseEmptyDivs(true);
				googletag.pubads().set("page_url", window.location.origin);
				googletag.pubads().addEventListener('slotRenderEnded', slotRendered);
				googletag.enableServices();
				dfpSlotsReady=true;
				
				DFPLogger.endTimer('page_dfp_init');
			} catch (error) {
				DFPLogger.error(`Failed to initialize ${page.name} DFP`, error);
			}
		});
	}
	else
		DFPLogger.info(`${page.name}: Using CM8 Ads (DFP disabled)`);
}

// Main DFP initialization function that determines page context and calls appropriate init function
// Uses strategy pattern to route to the correct initialization based on current scope
function initGlobalDfp(){
	pageTypeContext.initializeDfp();
}

window.aimDataLayer = window.aimDataLayer || [];

// AimTag helper function for medical audience targeting
// Pushes targeting data to the AimTag data layer for healthcare professional identification
function aimTag() { aimDataLayer.push(arguments); }

aimTag(AIMTAG_CONFIG.API_KEY, 'pageview');
console.debug('AimTag initialized');
/*
(function(w,d,s,m,n,t){
	w[m]=w[m]||{init:function(){(w[m].q=w[m].q||[]).push(arguments);},ready:function(c){if('function'!=typeof c){return;}(w[m].c=w[m].c||[]).push(c);c=w[m].c;
	n.onload=n.onreadystatechange=function(){if(!n.readyState||/loaded|complete/.test(n.readyState)){n.onload=n.onreadystatechange=null;
	if(t.parentNode&&n.parentNode){t.parentNode.removeChild(n);}while(c.length){(c.shift())();}}};}},w[m].d=1*new Date();n=d.createElement(s);t=d.getElementsByTagName(s)[0];
	n.async=1;n.src='//www.medtargetsystem.com/javascript/beacon.js?'+(Date.now().toString()).substring(0,5);n.setAttribute("data-aim",m);t.parentNode.insertBefore(n,t);
})(window,document,'script','AIM_126');

AIM_126.init('126-610-A5ECBB04');
console.debug('AIM_126 initialized');
*/