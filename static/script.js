/* particlesJS.load(@dom-id, @path-json, @callback (optional)); */
particlesJS.load('particles-js', 'static/particles.json', function() {
  console.log('callback - particles.js config loaded');
});

document.addEventListener('DOMContentLoaded', function () {
    
    // --- THE MAIN FIX ---
    // First, find the elements
    const searchForm = document.querySelector('#search-form');
    const micBtn = document.querySelector('#mic-btn');

    // **IMPORTANT**: If the form or button aren't on this specific page,
    // stop running the script to prevent errors.
    if (!searchForm || !micBtn) {
        console.log('Search form or mic button not found on this page.');
        return; // Stop right here
    }
    
    // If we get here, the elements exist.
    const searchInput = searchForm.querySelector('input[name="query"]');

    if ('webkitSpeechRecognition' in window) {
        const recognition = new webkitSpeechRecognition();
        recognition.continuous = false;
        
        // --- FEATURE 1: FASTER RESPONSE ---
        // Set to true to get live, in-progress results
        recognition.interimResults = true; 
        recognition.lang = 'en-US';

        // --- FEATURE 2: STOP ON CLICK (Implementation) ---
        // This function will be called if the page is clicked *while listening*
        const stopRecognitionOnClick = (event) => {
            // Check if the click was *not* on the mic button itself
            // We check the button and any icon inside it
            if (event.target !== micBtn && !micBtn.contains(event.target)) {
                console.log('Page clicked, stopping recognition.');
                recognition.stop();
                // The 'onend' event will handle removing this listener
            }
        };

        // Start recognition when the mic button is clicked
        micBtn.addEventListener('click', () => {
            try {
                recognition.start();
                // Add a "listening" class for visual feedback
                micBtn.classList.add('is-listening');
                // Add the "stop" listener to the body
                document.body.addEventListener('click', stopRecognitionOnClick);
            } catch (e) {
                console.error('Recognition could not start:', e);
                // Clean up if it fails to start
                micBtn.classList.remove('is-listening');
                document.body.removeEventListener('click', stopRecognitionOnClick);
            }
        });

        // Fired when speech recognition ends (for any reason)
        recognition.onend = () => {
            // ALWAYS clean up
            micBtn.classList.remove('is-listening');
            document.body.removeEventListener('click', stopRecognitionOnClick);
        };
        
        // Fired when an error happens
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            // ALWAYS clean up
            micBtn.classList.remove('is-listening');
            document.body.removeEventListener('click', stopRecognitionOnClick);
        };

        // --- Logic for interim (fast) results ---
        recognition.onresult = (event) => {
            let interimTranscript = '';
            let finalTranscript = '';

            for (let i = event.resultIndex; i < event.results.length; ++i) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript;
                } else {
                    interimTranscript += transcript;
                }
            }

            // Update the input field in real-time
            searchInput.value = finalTranscript || interimTranscript;

            // If we get a final result, submit the form
            if (finalTranscript) {
                searchInput.value = finalTranscript.trim();
                searchForm.submit();
            }
        };

    } else {
        // If not supported, hide the mic button
        micBtn.style.display = 'none';
        console.log('Speech recognition not supported in this browser.');
    }
});