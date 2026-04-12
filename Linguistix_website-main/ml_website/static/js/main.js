/**
 * Main JavaScript file for Speaker Recognition System
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize counters
    initCounters();

    // Reveal sections as they enter the viewport
    initRevealAnimations();

    // Add subtle 3D card tilt interactions
    initTiltCards();

    // Add magnetic cursor response for key buttons
    initMagneticButtons();

    // Drive hero orb motion from pointer movement
    initHeroOrbInteraction();

    // Show selected file name in upload block
    initUploadInputFeedback();
    
    // Add active class to current nav item
    setActiveNavItem();
    
    // Initialize any audio visualizations
    initAudioVisualizations();

    // Check if audio upload form exists on the page
    const uploadForm = document.getElementById('audio-upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleAudioUpload);
    }

    // Initialize any tooltips
    initTooltips();

    // Animate chart cards when entering viewport
    initChartCardPulse();
});

/**
 * Initialize counter animation for statistics
 */
function initCounters() {
    const counters = document.querySelectorAll('.counter');
    
    counters.forEach(counter => {
        const target = parseFloat(counter.getAttribute('data-target'));
        const duration = 1500; // ms
        const stepTime = 20; // ms
        const decimals = Number.isInteger(target) ? 0 : 2;
        
        let current = 0.0;
        const totalSteps = Math.max(1, Math.floor(duration / stepTime));
        const step = target / totalSteps;
        
        const timer = setInterval(() => {
            current += step;
            
            if (current > target) {
                counter.textContent = target.toFixed(decimals).replace(/\.00$/, '');
                clearInterval(timer);
            } else {
                counter.textContent = current.toFixed(decimals).replace(/\.00$/, '');
            }
        }, stepTime);
    });
}

/**
 * Reveal elements on scroll for staged entrance motion.
 */
function initRevealAnimations() {
    const revealElements = document.querySelectorAll('.reveal');
    if (!revealElements.length || !('IntersectionObserver' in window)) {
        revealElements.forEach(el => el.classList.add('revealed'));
        return;
    }

    const observer = new IntersectionObserver((entries, obs) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add('revealed');
            obs.unobserve(entry.target);
        });
    }, {
        threshold: 0.15,
        rootMargin: '0px 0px -40px 0px'
    });

    revealElements.forEach(el => observer.observe(el));
}

/**
 * Add pointer-based 3D tilt to cards.
 */
function initTiltCards() {
    const tiltCards = document.querySelectorAll('.tilt-card');

    tiltCards.forEach(card => {
        card.addEventListener('mousemove', (event) => {
            const rect = card.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            const rotateY = ((x / rect.width) - 0.5) * 10;
            const rotateX = (0.5 - (y / rect.height)) * 8;

            card.style.transform = `perspective(900px) rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg) translateY(-4px)`;
            card.style.transition = 'transform 0.08s linear';
        });

        card.addEventListener('mouseleave', () => {
            card.style.transform = '';
            card.style.transition = 'transform 0.35s ease';
        });
    });
}

/**
 * Make CTA buttons gently follow the cursor.
 */
function initMagneticButtons() {
    const magneticButtons = document.querySelectorAll('.btn-magnetic');

    magneticButtons.forEach(button => {
        button.addEventListener('mousemove', (event) => {
            const rect = button.getBoundingClientRect();
            const relX = event.clientX - rect.left - rect.width / 2;
            const relY = event.clientY - rect.top - rect.height / 2;
            const moveX = relX * 0.18;
            const moveY = relY * 0.22;
            button.style.transform = `translate(${moveX.toFixed(2)}px, ${moveY.toFixed(2)}px)`;
        });

        button.addEventListener('mouseleave', () => {
            button.style.transform = '';
        });
    });
}

/**
 * Interactive pseudo-3D hero orb controlled by pointer.
 */
function initHeroOrbInteraction() {
    const scene = document.getElementById('voice-orb-scene');
    if (!scene) return;

    const fallback = document.getElementById('voice-orb-fallback');
    const canvas = document.getElementById('voice-orb-canvas');
    const canUseThree = Boolean(window.THREE && canvas && !window.matchMedia('(prefers-reduced-motion: reduce)').matches);

    if (canUseThree && initThreeHeroOrb(scene, canvas, fallback)) {
        return;
    }

    const orb = document.getElementById('voice-orb');
    if (!orb) return;

    scene.addEventListener('mousemove', (event) => {
        const rect = scene.getBoundingClientRect();
        const x = (event.clientX - rect.left) / rect.width;
        const y = (event.clientY - rect.top) / rect.height;

        const rotateY = (x - 0.5) * 26;
        const rotateX = (0.5 - y) * 22;

        orb.style.transform = `rotateX(${rotateX.toFixed(2)}deg) rotateY(${rotateY.toFixed(2)}deg)`;
        scene.style.setProperty('--spot-x', `${(x * 100).toFixed(2)}%`);
        scene.style.setProperty('--spot-y', `${(y * 100).toFixed(2)}%`);
    });

    scene.addEventListener('mouseleave', () => {
        orb.style.transform = '';
        scene.style.setProperty('--spot-x', '50%');
        scene.style.setProperty('--spot-y', '50%');
    });
}

/**
 * Create a lightweight Three.js hero model with graceful fallback.
 * @param {HTMLElement} sceneEl
 * @param {HTMLCanvasElement} canvas
 * @param {HTMLElement|null} fallbackEl
 * @returns {boolean}
 */
function initThreeHeroOrb(sceneEl, canvas, fallbackEl) {
    try {
        const THREE = window.THREE;
        if (!THREE) return false;

        const renderer = new THREE.WebGLRenderer({
            canvas,
            antialias: true,
            alpha: true,
            powerPreference: 'high-performance'
        });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.8));

        const threeScene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
        camera.position.set(0, 0, 4.2);

        const group = new THREE.Group();
        threeScene.add(group);

        const core = new THREE.Mesh(
            new THREE.IcosahedronGeometry(1, 4),
            new THREE.MeshStandardMaterial({
                color: 0x8cb2ff,
                emissive: 0x1f3f8f,
                emissiveIntensity: 0.35,
                roughness: 0.32,
                metalness: 0.25,
                flatShading: true,
                transparent: true,
                opacity: 0.96
            })
        );

        const shell = new THREE.Mesh(
            new THREE.IcosahedronGeometry(1.18, 1),
            new THREE.MeshBasicMaterial({
                color: 0xa4c2ff,
                wireframe: true,
                transparent: true,
                opacity: 0.28
            })
        );

        const ringA = new THREE.Mesh(
            new THREE.TorusGeometry(1.55, 0.018, 16, 180),
            new THREE.MeshBasicMaterial({ color: 0x7ea8ff, transparent: true, opacity: 0.62 })
        );
        ringA.rotation.x = Math.PI * 0.45;

        const ringB = new THREE.Mesh(
            new THREE.TorusGeometry(1.68, 0.016, 16, 180),
            new THREE.MeshBasicMaterial({ color: 0x98beff, transparent: true, opacity: 0.5 })
        );
        ringB.rotation.y = Math.PI * 0.42;

        group.add(core);
        group.add(shell);
        group.add(ringA);
        group.add(ringB);

        const ambient = new THREE.AmbientLight(0x9ab7ff, 0.72);
        const keyLight = new THREE.DirectionalLight(0xaed1ff, 1.05);
        keyLight.position.set(2, 1.8, 2.6);
        const rimLight = new THREE.PointLight(0x4e73df, 1.3, 10);
        rimLight.position.set(-2.2, -1.2, 1.8);

        threeScene.add(ambient);
        threeScene.add(keyLight);
        threeScene.add(rimLight);

        const particleCount = 160;
        const particlePositions = new Float32Array(particleCount * 3);
        for (let i = 0; i < particleCount; i++) {
            const radius = 2.15 + Math.random() * 0.8;
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);

            particlePositions[i * 3 + 0] = radius * Math.sin(phi) * Math.cos(theta);
            particlePositions[i * 3 + 1] = radius * Math.cos(phi);
            particlePositions[i * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);
        }

        const particleGeometry = new THREE.BufferGeometry();
        particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));

        const particles = new THREE.Points(
            particleGeometry,
            new THREE.PointsMaterial({
                color: 0x9ec6ff,
                size: 0.028,
                transparent: true,
                opacity: 0.72
            })
        );
        threeScene.add(particles);

        const targetRotation = { x: 0, y: 0 };

        function resizeRenderer() {
            const bounds = sceneEl.getBoundingClientRect();
            const width = Math.max(1, Math.floor(bounds.width));
            const height = Math.max(1, Math.floor(bounds.height));

            renderer.setSize(width, height, false);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
        }

        resizeRenderer();
        if ('ResizeObserver' in window) {
            const resizeObserver = new ResizeObserver(() => resizeRenderer());
            resizeObserver.observe(sceneEl);
        } else {
            window.addEventListener('resize', resizeRenderer);
        }

        sceneEl.addEventListener('mousemove', (event) => {
            const rect = sceneEl.getBoundingClientRect();
            const nx = ((event.clientX - rect.left) / rect.width) - 0.5;
            const ny = ((event.clientY - rect.top) / rect.height) - 0.5;

            targetRotation.y = nx * 0.75;
            targetRotation.x = ny * 0.45;

            sceneEl.style.setProperty('--spot-x', `${((nx + 0.5) * 100).toFixed(2)}%`);
            sceneEl.style.setProperty('--spot-y', `${((ny + 0.5) * 100).toFixed(2)}%`);
        });

        sceneEl.addEventListener('mouseleave', () => {
            targetRotation.x = 0;
            targetRotation.y = 0;
            sceneEl.style.setProperty('--spot-x', '50%');
            sceneEl.style.setProperty('--spot-y', '50%');
        });

        const clock = new THREE.Clock();

        function animate() {
            const t = clock.getElapsedTime();

            group.rotation.y += (targetRotation.y - group.rotation.y) * 0.06;
            group.rotation.x += ((-targetRotation.x) - group.rotation.x) * 0.06;

            core.rotation.y = t * 0.3;
            core.rotation.x = t * 0.22;
            shell.rotation.y = -t * 0.2;
            ringA.rotation.z = t * 0.55;
            ringB.rotation.x = t * 0.48;
            particles.rotation.y = t * 0.08;
            particles.rotation.x = Math.sin(t * 0.4) * 0.14;

            const pulse = 1 + Math.sin(t * 2.4) * 0.035;
            core.scale.set(pulse, pulse, pulse);

            renderer.render(threeScene, camera);
            requestAnimationFrame(animate);
        }

        animate();

        if (fallbackEl) {
            fallbackEl.classList.add('is-hidden');
        }

        return true;
    } catch (error) {
        console.warn('Three.js hero initialization failed, using fallback.', error);
        if (fallbackEl) {
            fallbackEl.classList.remove('is-hidden');
        }
        return false;
    }
}

/**
 * Display selected file metadata under the upload input.
 */
function initUploadInputFeedback() {
    const input = document.getElementById('audio-file');
    const output = document.getElementById('selected-file-name');
    if (!input || !output) return;

    input.addEventListener('change', () => {
        const file = input.files && input.files[0] ? input.files[0] : null;
        if (!file) {
            output.textContent = '';
            output.classList.remove('show');
            return;
        }

        const fileSizeMb = (file.size / (1024 * 1024)).toFixed(2);
        output.textContent = `Selected: ${file.name} (${fileSizeMb} MB)`;
        output.classList.add('show');
    });
}

/**
 * Set active class on the current navigation item
 */
function setActiveNavItem() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar .nav-link');
    
    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        
        if (currentPath === linkPath || 
            (currentPath === '/' && linkPath === '/') || 
            (currentPath !== '/' && linkPath !== '/' && currentPath.includes(linkPath))) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

/**
 * Initialize audio visualizations for the audio player
 */
function initAudioVisualizations() {
    const audioPlayer = document.getElementById('audio-player');
    if (!audioPlayer) return;
    
    // Add event listener for the audio player
    audioPlayer.addEventListener('play', createVisualization);
}

/**
 * Create audio visualization if Web Audio API is available
 */
function createVisualization() {
    const audioPlayer = document.getElementById('audio-player');
    if (!audioPlayer) return;
    
    // Check for Web Audio API support
    if (!window.AudioContext && !window.webkitAudioContext) return;
    
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioContext();
    const analyser = audioCtx.createAnalyser();
    const source = audioCtx.createMediaElementSource(audioPlayer);
    
    source.connect(analyser);
    analyser.connect(audioCtx.destination);
    
    // Configure analyser
    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    // Remove event listener to avoid creating multiple analyzers
    audioPlayer.removeEventListener('play', createVisualization);
}

/**
 * Format time in seconds to MM:SS format
 * @param {number} seconds 
 * @returns {string} Formatted time
 */
function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds < 10 ? '0' : ''}${remainingSeconds}`;
}

/**
 * Handle file upload validation for audio files
 * @param {HTMLInputElement} fileInput 
 * @returns {boolean} Is valid file
 */
function validateAudioFile(fileInput) {
    if (!fileInput.files || fileInput.files.length === 0) {
        showError('Please select an audio file');
        return false;
    }
    
    const file = fileInput.files[0];
    const fileType = file.type;
    
    // Check if file is a WAV audio file
    if (!fileType.startsWith('audio/')) {
        showError('Please upload an audio file');
        return false;
    }
    
    if (file.size > 16 * 1024 * 1024) { // 16MB
        showError('File size exceeds maximum limit (16MB)');
        return false;
    }
    
    return true;
}

/**
 * Show error message
 * @param {string} message 
 */
function showError(message) {
    const resultArea = document.getElementById('result-area');
    if (resultArea) {
        resultArea.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <i class="fas fa-exclamation-circle me-2"></i> ${message}
            </div>
        `;
    }
}

/**
 * Handle audio file upload and prediction
 * @param {Event} event 
 */
function handleAudioUpload(event) {
    event.preventDefault();
    
    // Show loading spinner
    const resultArea = document.getElementById('result-area');
    if (resultArea) {
        resultArea.innerHTML = '<div class="text-center my-4"><div class="spinner-border text-primary" role="status"></div><p class="mt-2">Processing audio...</p></div>';
    }
    
    // Get form data
    const formData = new FormData(event.target);
    
    // Check if file is selected
    const audioFile = formData.get('audio_file');
    if (!audioFile || audioFile.size === 0) {
        showError('Please select an audio file');
        return;
    }
    
    // Make AJAX request
    fetch('/upload', {
        method: 'POST',
        body: formData,
        // Do not set Content-Type header when sending FormData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            displayPredictionResult(data);
        } else {
            showError(data.error || 'Unknown error occurred');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showError('Network error: ' + error.message);
    });
}

/**
 * Display prediction result in the UI
 * @param {Object} data 
 */
function displayPredictionResult(data) {
    const resultArea = document.getElementById('result-area');
    if (resultArea) {
        const confidenceRounded = Math.round((data.confidence || 0) * 10) / 10;
        const marginRounded = Math.round((data.confidence_margin || 0) * 10) / 10;
        const isUnknown = Boolean(data.is_unknown);

        const titleText = isUnknown ? 'Speaker Not Confidently Identified' : 'Speaker Identified!';
        const titleClass = isUnknown ? 'text-warning' : 'text-success';
        const titleIcon = isUnknown ? 'fa-triangle-exclamation' : 'fa-check-circle';

        let closestMatchHtml = '';
        if (isUnknown && data.matched_speaker) {
            closestMatchHtml = `<strong>Closest match:</strong> ${data.matched_speaker}<br>`;
        }

        let topPredictionsHtml = '';
        if (Array.isArray(data.top_predictions) && data.top_predictions.length > 0) {
            const rows = data.top_predictions
                .map(item => {
                    const conf = Math.round((item.confidence || 0) * 10) / 10;
                    return `<li class="top-prediction-item">#${item.rank} ${item.speaker} <span>${conf}%</span></li>`;
                })
                .join('');

            topPredictionsHtml = `
                <div class="top-predictions mt-3">
                    <strong>Top predictions:</strong>
                    <ul class="top-prediction-list mt-2 mb-0">${rows}</ul>
                </div>
            `;
        }

        resultArea.innerHTML = `
            <div class="card shadow-sm">
                <div class="card-body">
                    <h5 class="card-title ${titleClass}"><i class="fas ${titleIcon} me-2"></i>${titleText}</h5>
                    <p class="card-text">
                        <strong>Speaker identified as:</strong> ${data.speaker}<br>
                        ${closestMatchHtml}
                        <strong>Confidence score:</strong> ${confidenceRounded}%<br>
                        <strong>Top-1 margin vs #2:</strong> ${marginRounded}%<br>
                        <strong>Decision:</strong> ${data.decision_reason || 'accepted'}
                    </p>
                    ${topPredictionsHtml}
                </div>
            </div>
        `;
        
        // Log the actual data received for debugging
        console.log('Prediction data received:', data);
    }
}

/**
 * Initialize Bootstrap tooltips
 */
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    if (tooltipTriggerList.length > 0) {
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

/**
 * Add a subtle one-time pulse when chart cards enter viewport.
 */
function initChartCardPulse() {
    const chartCards = document.querySelectorAll('.chart-container');
    if (!chartCards.length || !('IntersectionObserver' in window)) return;

    const observer = new IntersectionObserver((entries, obs) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            entry.target.classList.add('chart-activated');
            obs.unobserve(entry.target);
        });
    }, {
        threshold: 0.18
    });

    chartCards.forEach(card => observer.observe(card));
}