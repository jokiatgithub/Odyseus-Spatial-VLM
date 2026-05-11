import * as THREE from "three";
import { OrbitControls } from "https://unpkg.com/three@0.165.0/examples/jsm/controls/OrbitControls.js";

(function () {
  const form = document.getElementById("demo-form");
  const imageInput = document.getElementById("image");
  const uploadPreview = document.getElementById("upload-preview");
  const latestPreview = document.getElementById("latest-preview");
  const latestPlaceholder = document.getElementById("latest-placeholder");
  const fileNameEl = document.getElementById("file-name");
  const depthPreview = document.getElementById("depth-preview");
  const depthPlaceholder = document.getElementById("depth-placeholder");
  const annotatedPreview = document.getElementById("annotated-preview");
  const annotatedPlaceholder = document.getElementById("annotated-placeholder");
  const statusEl = document.getElementById("status");
  const metaEl = document.getElementById("meta");
  const viewerEl = document.getElementById("viewer");
  const submitButton = document.getElementById("submit");
  const promptInput = document.getElementById("prompt");
  const progressPercent = document.getElementById("progress-percent");
  const progressFill = document.getElementById("progress-fill");
  const progressCaption = document.getElementById("progress-caption");
  const sessionIdEl = document.getElementById("session-id");
  const runStateEl = document.getElementById("run-state");
  const frameCountEl = document.getElementById("frame-count");
  const sourceLabelEl = document.getElementById("source-label");
  const promptStateEl = document.getElementById("prompt-state");
  const requestPayloadEl = document.getElementById("request-payload");
  const responsePayloadEl = document.getElementById("response-payload");

  let previewUrl = null;
  let activeSessionId = "None";
  let pointsObject = null;
  let markerData = [];
  let markerSpheres = [];
  let cameraGroup = null;

  const PUBLIC_API_ORIGIN = "http://178.105.98.166:8080";
  const POINT_CLOUD_POINT_SIZE = 0.034;
  const POINT_CLOUD_CAMERA_DISTANCE = 1.9;
  const POINT_COLOR_EXPOSURE = 1.08;
  const POINT_COLOR_SATURATION = 1.32;
  const POINT_COLOR_GAMMA = 0.88;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090b0f);

  const camera = new THREE.PerspectiveCamera(
    60,
    viewerEl.clientWidth / Math.max(viewerEl.clientHeight, 1),
    0.01,
    1000
  );
  camera.position.set(0, 0, 4);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(viewerEl.clientWidth, Math.max(viewerEl.clientHeight, 1));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NoToneMapping;
  viewerEl.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0, -1);

  const grid = new THREE.GridHelper(4, 12, 0x2f3a46, 0x1a222c);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -2;
  scene.add(grid);

  const markerGroup = new THREE.Group();
  scene.add(markerGroup);

  const MARKER_COLORS = {
    chair: 0xff4444,
    table: 0x44ff44,
    door: 0x4444ff,
    person: 0xff8800,
    plant: 0x00cc44,
    monitor: 0x00ccff,
    lamp: 0xffff00,
    window: 0x88ccff,
    couch: 0xcc44cc,
    bed: 0xff6688,
    sink: 0x44cccc,
    toilet: 0xcccc44,
    tv: 0x0088ff,
    book: 0xcc8844,
    bottle: 0x44ccaa,
    cup: 0xffaa44,
    keyboard: 0xaaaaaa,
    phone: 0x88ff88,
    shelf: 0x886644,
    box: 0xff44aa,
    cabinet: 0x668844,
  };
  const TARGET_PALETTE = [
    0xff4444,
    0x44ff44,
    0x4444ff,
    0xff8800,
    0x00ccff,
    0xcc44cc,
    0xffff00,
    0x44cccc,
  ];

  const tooltip = document.createElement("div");
  tooltip.style.cssText =
    "position:fixed;padding:8px 12px;background:rgba(0,0,0,0.85);color:#fff;" +
    "border-radius:6px;font-size:12px;pointer-events:none;display:none;z-index:999;" +
    "font-family:monospace;border:1px solid rgba(255,255,255,0.2);max-width:250px;";
  document.body.appendChild(tooltip);

  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function makeSessionId() {
    if (window.crypto?.randomUUID) {
      return window.crypto.randomUUID();
    }
    return `${Date.now().toString(16)}-${Math.random().toString(16).slice(2, 10)}`;
  }

  function formatSessionId(sessionId) {
    if (!sessionId || sessionId === "None") {
      return "None";
    }
    return sessionId.replaceAll("-", "-\n");
  }

  function setProgress(percent, caption) {
    progressPercent.textContent = `${percent}%`;
    progressFill.style.width = `${percent}%`;
    progressCaption.textContent = caption;
  }

  function setMetricState({ status, frameCount, source, promptState }) {
    if (status !== undefined) runStateEl.textContent = status;
    if (frameCount !== undefined) frameCountEl.textContent = String(frameCount);
    if (source !== undefined) sourceLabelEl.textContent = source;
    if (promptState !== undefined) promptStateEl.textContent = promptState;
  }

  function setPayload(el, payload) {
    el.textContent = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
  }

  let resolvedApiBase = null;

  function apiUrl(base, path) {
    return base ? `${base}${path}` : path;
  }

  async function resolveApiBase() {
    if (resolvedApiBase !== null) {
      return resolvedApiBase;
    }

    const candidates = ["", PUBLIC_API_ORIGIN];
    const errors = [];
    for (const base of [...new Set(candidates)]) {
      const endpoint = apiUrl(base, "/healthz");
      try {
        const response = await fetch(endpoint, { cache: "no-store" });
        if (response.ok) {
          resolvedApiBase = base;
          return resolvedApiBase;
        }
        errors.push(`${endpoint} -> HTTP ${response.status}`);
      } catch (error) {
        errors.push(`${endpoint} -> ${error}`);
      }
    }

    throw new Error(
      `Cannot reach the Blinkin VLM API from this browser page. Current page: ${window.location.href}. Tried: ${errors.join("; ")}`
    );
  }

  async function apiFetch(path, options = {}) {
    const base = await resolveApiBase();
    return fetch(apiUrl(base, path), options);
  }

  function sleep(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
  }

  async function runInferJob(formData) {
    const createResponse = await apiFetch("/api/jobs", {
      method: "POST",
      body: formData,
    });
    const createPayload = await createResponse.json().catch(() => ({}));
    if (!createResponse.ok) {
      throw new Error(createPayload.detail || `Failed to create inference job (${createResponse.status}).`);
    }

    const jobId = createPayload.job_id;
    setPayload(responsePayloadEl, {
      job_id: jobId,
      status: "queued",
      message: "Single-image inference queued.",
    });

    for (let attempt = 0; attempt < 180; attempt += 1) {
      await sleep(attempt < 4 ? 1000 : 2500);
      const jobResponse = await apiFetch(`/api/jobs/${jobId}`, {
        cache: "no-store",
      });
      const job = await jobResponse.json().catch(() => ({}));
      if (!jobResponse.ok) {
        throw new Error(job.detail || `Failed to read inference job (${jobResponse.status}).`);
      }

      if (job.status === "queued") {
        setProgress(50, job.message || "Queued single-image inference");
        setPayload(responsePayloadEl, job);
        continue;
      }

      if (job.status === "running") {
        setProgress(Math.min(95, 55 + attempt), job.message || "Running Blinkin VLM");
        setPayload(responsePayloadEl, job);
        continue;
      }

      if (job.status === "failed") {
        throw new Error(job.error || "Inference failed.");
      }

      if (job.status === "succeeded") {
        return job.result;
      }
    }

    throw new Error("Inference job timed out while waiting for the remote model response.");
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => {
      const entities = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      };
      return entities[char];
    });
  }

  function setPreviewImage(imageEl, placeholderEl, src, fallbackText) {
    if (src) {
      imageEl.src = src;
      imageEl.hidden = false;
      placeholderEl.hidden = true;
      return;
    }

    imageEl.removeAttribute("src");
    imageEl.hidden = true;
    placeholderEl.textContent = fallbackText;
    placeholderEl.hidden = false;
  }

  function setPreviewLoading(promptValue) {
    setPreviewImage(depthPreview, depthPlaceholder, "", "Building scene...");
    setPreviewImage(
      annotatedPreview,
      annotatedPlaceholder,
      "",
      promptValue ? "Locating targets..." : "No prompt targets."
    );
  }

  function resetResultPreviews() {
    setPreviewImage(depthPreview, depthPlaceholder, "", "No scene yet.");
    setPreviewImage(annotatedPreview, annotatedPlaceholder, "", "No targets yet.");
  }

  function renderTargetList(targets) {
    if (!targets.length) {
      metaEl.textContent = "No targets yet.";
      return;
    }
    metaEl.innerHTML = targets
      .map((target) => {
        const color = `#${getMarkerColor(target.label, target.colorIndex).toString(16).padStart(6, "0")}`;
        const coords = `${target.position.x.toFixed(2)}, ${target.position.y.toFixed(2)}, ${target.position.z.toFixed(2)}`;
        return `
          <div class="target-item">
            <span class="target-swatch" style="background:${color}"></span>
            <span>${escapeHtml(target.label)}  (${coords})</span>
          </div>
        `;
      })
      .join("");
  }

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  function resize() {
    const width = viewerEl.clientWidth;
    const height = Math.max(viewerEl.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  function resetPointCloud() {
    if (!pointsObject) {
      resetMarkers();
      return;
    }
    scene.remove(pointsObject);
    pointsObject.geometry.dispose();
    pointsObject.material.dispose();
    pointsObject = null;
    resetMarkers();
    resetCameraContext();
  }

  function getMarkerColor(label, colorIndex = 0) {
    const base = label.replace(/_\d+$/, "").replace(/ \d+$/, "");
    return MARKER_COLORS[base] || TARGET_PALETTE[colorIndex % TARGET_PALETTE.length];
  }

  function clamp01(value) {
    return Math.min(1, Math.max(0, value));
  }

  function gradePointColors(colors) {
    const graded = [];
    for (const color of colors) {
      const [r, g, b] = color;
      const luma = r * 0.2126 + g * 0.7152 + b * 0.0722;
      const channels = [r, g, b];
      for (const channel of channels) {
        const saturated = luma + (channel - luma) * POINT_COLOR_SATURATION;
        const exposed = Math.pow(clamp01(saturated), POINT_COLOR_GAMMA) * POINT_COLOR_EXPOSURE;
        graded.push(clamp01(exposed));
      }
    }
    return graded;
  }

  function createTextSprite(text, color, subtitle) {
    const canvas = document.createElement("canvas");
    const width = 256;
    const height = subtitle ? 100 : 70;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "rgba(0,0,0,0.8)";
    ctx.beginPath();
    ctx.roundRect(4, 4, width - 8, height - 8, 10);
    ctx.fill();

    const hex = `#${color.toString(16).padStart(6, "0")}`;
    ctx.strokeStyle = hex;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(4, 4, width - 8, height - 8, 10);
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 26px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text.toUpperCase(), width / 2, subtitle ? 30 : height / 2);

    if (subtitle) {
      ctx.fillStyle = "#aaaaaa";
      ctx.font = "18px sans-serif";
      ctx.fillText(subtitle, width / 2, 65);
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.minFilter = THREE.LinearFilter;
    const material = new THREE.SpriteMaterial({
      map: texture,
      depthTest: false,
      sizeAttenuation: true,
    });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(0.35, subtitle ? 0.14 : 0.1, 1);
    return sprite;
  }

  function resetMarkers() {
    markerData = [];
    markerSpheres = [];
    while (markerGroup.children.length) {
      const child = markerGroup.children[0];
      markerGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (child.material.map) child.material.map.dispose();
        child.material.dispose();
      }
    }
    tooltip.style.display = "none";
  }

  function resetCameraContext() {
    if (!cameraGroup) {
      return;
    }
    while (cameraGroup.children.length) {
      const child = cameraGroup.children[0];
      cameraGroup.remove(child);
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
      if (child.line) {
        child.line.geometry.dispose();
        child.line.material.dispose();
      }
      if (child.cone) {
        child.cone.geometry.dispose();
        child.cone.material.dispose();
      }
    }
    scene.remove(cameraGroup);
    cameraGroup = null;
  }

  function updateCameraContext(markers) {
    resetCameraContext();
    cameraGroup = new THREE.Group();

    const frustumDepth = 0.28;
    const frustumHalfW = 0.18;
    const frustumHalfH = 0.12;
    const frustumPoints = [
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(-frustumHalfW, frustumHalfH, -frustumDepth),
      new THREE.Vector3(frustumHalfW, frustumHalfH, -frustumDepth),
      new THREE.Vector3(frustumHalfW, -frustumHalfH, -frustumDepth),
      new THREE.Vector3(-frustumHalfW, -frustumHalfH, -frustumDepth),
    ];
    const frustumIndices = [
      [0, 1], [0, 2], [0, 3], [0, 4],
      [1, 2], [2, 3], [3, 4], [4, 1],
    ];
    for (const [a, b] of frustumIndices) {
      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([frustumPoints[a], frustumPoints[b]]),
        new THREE.LineBasicMaterial({ color: 0x7cc7ff, transparent: true, opacity: 0.95 })
      );
      cameraGroup.add(line);
    }

    const radius = pointsObject?.geometry?.boundingSphere?.radius || 1;
    const defaultArrowLength = Math.max(0.1, radius * 0.06);
    for (const marker of markers) {
      const color = getMarkerColor(marker.label, marker.colorIndex);
      const target = new THREE.Vector3(
        marker.position.x,
        marker.position.y,
        marker.position.z
      );
      const distance = target.length();
      if (distance < 1e-8) continue;
      const arrowLength = Math.min(defaultArrowLength, distance);
      const dir = target.clone().normalize();

      const ray = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(0, 0, 0),
          target.clone(),
        ]),
        new THREE.LineDashedMaterial({
          color,
          transparent: true,
          opacity: 0.45,
          dashSize: 0.035,
          gapSize: 0.03,
        })
      );
      ray.computeLineDistances();
      cameraGroup.add(ray);

      const headLength = Math.min(0.06, arrowLength * 0.32);
      const shaftLength = Math.max(arrowLength - headLength, 0.02);
      const shaftRadius = 0.018;
      const headRadius = 0.045;
      const material = new THREE.MeshBasicMaterial({ color });

      const shaft = new THREE.Mesh(
        new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLength, 12),
        material
      );
      shaft.position.copy(dir.clone().multiplyScalar(shaftLength * 0.5));
      shaft.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir
      );
      cameraGroup.add(shaft);

      const cone = new THREE.Mesh(
        new THREE.ConeGeometry(headRadius, headLength, 12),
        material.clone()
      );
      cone.position.copy(dir.clone().multiplyScalar(shaftLength + headLength * 0.5));
      cone.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir
      );
      cameraGroup.add(cone);
    }

    scene.add(cameraGroup);
  }

  function updateMarkers(markers) {
    resetMarkers();
    markerData = markers;

    for (const marker of markers) {
      const { x, y, z } = marker.position;
      const color = getMarkerColor(marker.label, marker.colorIndex);
      const coords = `(${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)})`;

      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.04),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.92 })
      );
      sphere.position.set(x, y, z);
      sphere.userData = { marker, coords };
      markerGroup.add(sphere);
      markerSpheres.push(sphere);

      const stem = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(x, y, z),
          new THREE.Vector3(x, y + 0.18, z),
        ]),
        new THREE.LineBasicMaterial({ color })
      );
      markerGroup.add(stem);

      const sprite = createTextSprite(
        marker.label,
        color,
        `${coords} ${(marker.confidence * 100).toFixed(0)}%`
      );
      sprite.position.set(x, y + 0.28, z);
      markerGroup.add(sprite);
    }
  }

  function loadPointCloud(points, colors) {
    resetPointCloud();

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(points.flat(), 3)
    );
    geometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(gradePointColors(colors), 3)
    );

    const material = new THREE.PointsMaterial({
      size: POINT_CLOUD_POINT_SIZE,
      vertexColors: true,
      sizeAttenuation: true,
    });

    pointsObject = new THREE.Points(geometry, material);
    scene.add(pointsObject);

    geometry.computeBoundingSphere();
    const sphere = geometry.boundingSphere;
    if (!sphere) {
      return;
    }

    controls.target.copy(sphere.center);
    const radius = Math.max(sphere.radius, 0.25);
    camera.position.set(
      sphere.center.x,
      sphere.center.y,
      sphere.center.z + radius * POINT_CLOUD_CAMERA_DISTANCE
    );
    camera.near = Math.max(radius / 500, 0.01);
    camera.far = Math.max(radius * 20, 100);
    camera.updateProjectionMatrix();
    controls.update();
  }

  imageInput.addEventListener("change", () => {
    const file = imageInput.files && imageInput.files[0];
    if (!file) {
      return;
    }
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    previewUrl = URL.createObjectURL(file);
    fileNameEl.textContent = file.name;
    uploadPreview.src = previewUrl;
    uploadPreview.hidden = false;
    latestPreview.src = previewUrl;
    latestPreview.hidden = false;
    latestPlaceholder.hidden = true;
    activeSessionId = makeSessionId();
    sessionIdEl.textContent = formatSessionId(activeSessionId);
    setMetricState({
      status: "Image selected",
      frameCount: 0,
      source: "local image",
      promptState: promptInput.value.trim() ? "Set" : "Unset",
    });
    setProgress(0, "Idle, ready for frame processing");
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = imageInput.files && imageInput.files[0];
    if (!file) {
      setStatus("Choose an image first.");
      return;
    }

    if (!previewUrl) {
      previewUrl = URL.createObjectURL(file);
    }

    if (!activeSessionId || activeSessionId === "None") {
      activeSessionId = makeSessionId();
    }

    submitButton.disabled = true;
    setStatus("Running Blinkin VLM...");
    metaEl.textContent = "Running...";
    resetPointCloud();

    const formData = new FormData();
    formData.append("image", file);
    const promptValue = promptInput.value.trim();
    formData.append("prompt", promptValue);
    const requestPayload = {
      prompt: promptValue || null,
      file: file.name,
      input_type: "image",
      point_cloud_model: "depth-anything-v3",
      max_dist: 50,
      robot_height: 0.1,
      robot_radius: 0.15,
    };
    sessionIdEl.textContent = formatSessionId(activeSessionId);
    setMetricState({
      status: "Uploading single image",
      frameCount: 1,
      source: "local image",
      promptState: promptValue ? "Running" : "Unset",
    });
    setProgress(45, "Processing image: waiting for mesh and payload");
    setPayload(requestPayloadEl, requestPayload);
    setPayload(responsePayloadEl, "Opening single-image streams...");
    setPreviewLoading(promptValue);

    try {
      const result = await runInferJob(formData);

      setPreviewImage(
        depthPreview,
        depthPlaceholder,
        result.depth_preview ? `data:image/png;base64,${result.depth_preview}` : "",
        "Scene unavailable."
      );
      setPreviewImage(
        annotatedPreview,
        annotatedPlaceholder,
        result.annotated_preview ? `data:image/png;base64,${result.annotated_preview}` : "",
        result.meta?.target_error ? "Targets unavailable." : "No targets found."
      );
      const targets = (result.targets_3d || []).map((target, index) => ({
        ...target,
        colorIndex: index,
      }));
      renderTargetList(targets);
      loadPointCloud(result.points, result.colors);
      updateMarkers(targets);
      updateCameraContext(targets);
      const targetError = result.meta.target_error;
      setProgress(100, "Image processed: mesh and payload ready");
      setMetricState({
        status: targetError ? "Target error" : "ok",
        frameCount: 1,
        source: "single-infer",
        promptState: targetError ? "error" : promptValue ? "ok" : "Unset",
      });
      setPayload(responsePayloadEl, {
        single_infer: true,
        session_id: activeSessionId,
        max_depth_m: result.meta.depth_far_m,
        status: targetError ? "target_error" : "ok",
        objective_status: targetError ? "pending" : "complete",
        slam: {
          status: "ok",
          frames_processed: 1,
          has_mesh: Boolean(result.points?.length),
        },
        objects: targets.map((target) => ({
          label: target.label,
          confidence: target.confidence,
          pixel: target.pixel,
          position: target.position,
        })),
        error: targetError || null,
      });
      setStatus(
        targetError
          ? `Rendered ${result.meta.point_count} points. ${targetError}`
          : `Rendered ${result.meta.point_count} points and ${result.meta.target_count || 0} prompt targets.`
      );
    } catch (error) {
      setStatus(String(error));
      metaEl.textContent = "No targets yet.";
      setProgress(0, "Frame processing failed");
      setMetricState({
        status: "error",
        frameCount: 0,
        source: "local image",
        promptState: "error",
      });
      setPayload(responsePayloadEl, { status: "error", error: String(error) });
      resetResultPreviews();
      resetPointCloud();
    } finally {
      submitButton.disabled = false;
    }
  });

  renderer.domElement.addEventListener("mousemove", (event) => {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const hits = raycaster.intersectObjects(markerSpheres);

    if (!hits.length) {
      tooltip.style.display = "none";
      renderer.domElement.style.cursor = "default";
      return;
    }

    const { marker, coords } = hits[0].object.userData;
    tooltip.innerHTML =
      `<b>${marker.label.toUpperCase()}</b><br>` +
      `Position: ${coords}<br>` +
      `Confidence: ${(marker.confidence * 100).toFixed(0)}%<br>` +
      `Pixel: (${marker.pixel.x}, ${marker.pixel.y})`;
    tooltip.style.display = "block";
    tooltip.style.left = `${event.clientX + 14}px`;
    tooltip.style.top = `${event.clientY + 14}px`;
    renderer.domElement.style.cursor = "pointer";
  });

  window.addEventListener("resize", resize);
  resize();
  animate();
})();
