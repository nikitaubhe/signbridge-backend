/**
 * AvatarRenderer - Loads and animates the Blender-exported Remy 3D avatar (.glb files).
 * Uses ES modules for Three.js r160+ with GLTFLoader and OrbitControls.
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';

const AvatarRenderer = (function () {
    let scene, camera, renderer, mixer, clock, controls;
    let currentModel = null;
    let container = null;
    let isInitialized = false;
    let isLoading = false;
    const loader = new GLTFLoader();
    loader.setMeshoptDecoder(MeshoptDecoder);

    // Map sign phrases to .glb file names
    const SIGN_MODEL_MAP = {
        'hello': 'Hello.glb',
        'hi': 'Hi.glb',
        'thank you': 'Thank You.glb',
        'good evening': 'Good evening.glb',
        'how are u': 'How are u.glb',
        'i am fine': 'I am fine.glb',
        'i need water': 'I need water.glb'
    };

    const MODELS_PATH = '/api/models/';

    function init() {
        container = document.getElementById('avatar-canvas-container');
        if (!container || isInitialized) return;

        // Scene setup
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111827);

        // Subtle floor
        const floorGeo = new THREE.PlaneGeometry(20, 20);
        const floorMat = new THREE.MeshStandardMaterial({
            color: 0x1e293b,
            roughness: 0.9,
            metalness: 0.1
        });
        const floor = new THREE.Mesh(floorGeo, floorMat);
        floor.rotation.x = -Math.PI / 2;
        floor.position.y = 0;
        floor.receiveShadow = true;
        scene.add(floor);

        // Camera
        const width = container.clientWidth || 800;
        const height = container.clientHeight || 450;
        camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
        camera.position.set(0, 1.0, 4.0);
        camera.lookAt(0, 0.9, 0);

        // Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.outputColorSpace = THREE.SRGBColorSpace;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.2;
        container.appendChild(renderer.domElement);

        // Lighting
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
        keyLight.position.set(3, 5, 3);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 1024;
        keyLight.shadow.mapSize.height = 1024;
        scene.add(keyLight);

        const fillLight = new THREE.DirectionalLight(0x8b9dc3, 0.4);
        fillLight.position.set(-3, 3, -2);
        scene.add(fillLight);

        const rimLight = new THREE.DirectionalLight(0xa78bfa, 0.6);
        rimLight.position.set(0, 3, -4);
        scene.add(rimLight);

        // Grid
        const grid = new THREE.GridHelper(10, 20, 0x334155, 0x1e293b);
        grid.position.y = 0.001;
        scene.add(grid);

        clock = new THREE.Clock();

        // Orbit Controls
        controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set(0, 0.8, 0);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance = 0.5;
        controls.maxDistance = 1000;
        controls.maxPolarAngle = Math.PI / 2;
        controls.update();

        // Handle window resize
        window.addEventListener('resize', handleResize);

        isInitialized = true;
        animate();
        console.log('Avatar Renderer Initialized (ES Module)');

        // Load the idle/base model
        loadIdleModel();
    }

    function handleResize() {
        if (!container || !camera || !renderer) return;
        const width = container.clientWidth;
        const height = container.clientHeight;
        if (width === 0 || height === 0) return;

        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderer.setSize(width, height);
    }

    function showLoading(msg) {
        const statusEl = document.getElementById('avatar-status');
        if (statusEl) {
            statusEl.textContent = msg || 'Loading...';
            statusEl.style.color = '#f59e0b';
        }
        let spinner = container.querySelector('.avatar-loading-overlay');
        if (!spinner) {
            spinner = document.createElement('div');
            spinner.className = 'avatar-loading-overlay';
            spinner.innerHTML = '<div class="avatar-spinner"></div><p>Loading model...</p>';
            container.appendChild(spinner);
        }
        spinner.querySelector('p').textContent = msg || 'Loading model...';
        spinner.style.display = 'flex';
    }

    function hideLoading() {
        const spinner = container ? container.querySelector('.avatar-loading-overlay') : null;
        if (spinner) spinner.style.display = 'none';
    }

    function updateStatus(msg, type) {
        const statusEl = document.getElementById('avatar-status');
        if (statusEl) {
            statusEl.textContent = msg;
            if (type === 'success') statusEl.style.color = '#10b981';
            else if (type === 'error') statusEl.style.color = '#ef4444';
            else statusEl.style.color = '#94a3b8';
        }
    }

    function clearCurrentModel() {
        if (mixer) {
            mixer.stopAllAction();
            mixer.uncacheRoot(mixer.getRoot());
            mixer = null;
        }
        if (currentModel) {
            scene.remove(currentModel);
            currentModel.traverse((child) => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) {
                    if (Array.isArray(child.material)) {
                        child.material.forEach(m => m.dispose());
                    } else {
                        child.material.dispose();
                    }
                }
            });
            currentModel = null;
        }
    }

    function loadModel(filename, onLoaded) {
        const url = MODELS_PATH + encodeURIComponent(filename);
        console.log('Loading model from:', url);

        loader.load(
            url,
            (gltf) => {
                console.log(`Model loaded: ${filename}`, gltf);
                onLoaded(gltf);
            },
            (progress) => {
                if (progress.total > 0) {
                    const pct = Math.round((progress.loaded / progress.total) * 100);
                    showLoading(`Loading ${filename}... ${pct}%`);
                }
            },
            (error) => {
                console.error(`Error loading ${filename}:`, error);
                hideLoading();
                updateStatus(`Failed to load model`, 'error');
                isLoading = false;
            }
        );
    }

    function addModelToScene(gltf) {
        const model = gltf.scene;
        model.scale.set(1, 1, 1);
        model.position.set(0, 0, 0);

        model.traverse((child) => {
            if (child.isMesh) {
                child.castShadow = true;
                child.receiveShadow = true;
            }
        });

        scene.add(model);
        currentModel = model;
        fitModelToCamera(model);
        return model;
    }

    function loadIdleModel() {
        showLoading('Loading Remy Avatar...');
        loadModel('Remy.glb', (gltf) => {
            clearCurrentModel();
            addModelToScene(gltf);

            // Play idle animation if present
            if (gltf.animations && gltf.animations.length > 0) {
                mixer = new THREE.AnimationMixer(currentModel);
                const idleAction = mixer.clipAction(gltf.animations[0]);
                idleAction.play();
            }

            hideLoading();
            updateStatus('Avatar Ready', 'success');
            console.log('Remy idle model loaded');
        });
    }

    function fitModelToCamera(model) {
        console.log("Applying completely hardcoded avatar parameters as requested...");

        // 1. HARDCODE SCALE: 0.9 scale to look prominent
        model.scale.set(0.9, 0.9, 0.9);

        // 2. HARDCODE POSITION: -3.8 was underneath the floor! Bring it up significantly.
        model.position.set(0, -1.0, 0);

        // 3. HARDCODE CAMERA: Pull the camera much closer and slightly up
        camera.position.set(0, 1.0, 6.0);
        camera.lookAt(0, 0, 0);

        // 4. ELIMINATE CLIPPING: Make sure it cannot vanish due to being too close or too far
        camera.near = 0.01;
        camera.far = 10000;
        camera.updateProjectionMatrix();

        // 5. LOCK CONTROLS: Lock target and prevent panning
        if (controls) {
            controls.target.set(0, 0, 0);
            controls.minDistance = 1;
            controls.maxDistance = 20;
            controls.enablePan = false;
            controls.update();
        }
    }

    function playAnimation(signName) {
        if (isLoading) return;

        const key = signName.toLowerCase().trim();
        const filename = SIGN_MODEL_MAP[key];

        if (!filename) {
            console.warn(`No animation found for: "${signName}"`);
            updateStatus(`No animation for "${signName}"`, 'error');
            return;
        }

        console.log(`Playing animation: ${signName} -> ${filename}`);
        isLoading = true;
        
        // Auto-scroll to avatar container
        const avatarContainer = document.querySelector('.avatar-container');
        if (avatarContainer) {
            avatarContainer.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }

        showLoading(`Performing: ${signName}...`);

        loadModel(filename, (gltf) => {
            clearCurrentModel();
            addModelToScene(gltf);

            if (gltf.animations && gltf.animations.length > 0) {
                mixer = new THREE.AnimationMixer(currentModel);
                const clip = gltf.animations[0];
                const action = mixer.clipAction(clip);
                action.setLoop(THREE.LoopOnce);
                action.clampWhenFinished = true;
                action.play();

                mixer.addEventListener('finished', () => {
                    console.log(`Animation "${signName}" finished`);
                    setTimeout(() => loadIdleModel(), 500);
                });

                hideLoading();
                updateStatus(`Performing: ${signName}`, 'success');
            } else {
                // Static pose — show for 3 seconds then return to idle
                hideLoading();
                updateStatus(`Showing: ${signName}`, 'success');
                setTimeout(() => loadIdleModel(), 3000);
            }

            isLoading = false;
        });
    }

    function animate() {
        if (!isInitialized) return;
        requestAnimationFrame(animate);

        const delta = clock.getDelta();
        if (mixer) mixer.update(delta);
        if (controls) controls.update();

        renderer.render(scene, camera);
    }

    // Initialize on page load
    window.addEventListener('load', init);

    return {
        init: init,
        handleResize: handleResize,
        playAnimation: playAnimation
    };
})();

window.AvatarRenderer = AvatarRenderer;
