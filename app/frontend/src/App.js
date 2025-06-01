// app/frontend/src/App.js
import React, { useState, useEffect, useRef } from 'react'; // Added useRef
import './App.css';
import ImageUploadForm from './ImageUploadForm';
import GenerationControls from './GenerationControls';
// import OutputDisplay from './OutputDisplay'; // We will create this

const BACKEND_URL = "http://localhost:8000";
const POLLING_INTERVAL = 3000; // 3 seconds

// New OutputDisplay component
function OutputDisplay({ result, backendUrl }) {
  if (!result.outputImageId && !result.isLoading && !result.error) return null;

  return (
    <div className="output-area">
      <h3>Generation Result</h3>
      {result.isLoading && <p>Status: {result.statusMessage || 'Processing...'}</p>}
      {result.error && <p className="error-message">Error: {result.error}</p>}

      {result.imageUrl && (
        <div className="generated-image-container">
          <img src={result.imageUrl} alt="Generated Output" style={{ maxWidth: '100%', maxHeight: '400px' }} />
          <div>
            <a href={result.imageUrl} target="_blank" rel="noopener noreferrer" download={`generated_image_${result.outputImageId}.png`} style={{marginRight: '10px'}}>
              Download Image (.png)
            </a>
            <a href={result.metadataUrl} target="_blank" rel="noopener noreferrer" download={`metadata_${result.outputImageId}.json`} style={{marginRight: '10px'}}>
              Download Metadata (.json)
            </a>
            <a href={`${backendUrl}/download_output_zip/${result.outputImageId}`}
               target="_blank" rel="noopener noreferrer" download={`output_${result.outputImageId}.zip`}>
              Download All (.zip)
            </a>
          </div>
        </div>
      )}
    </div>
  );
}


function App() {
  const [uploadedFileIds, setUploadedFileIds] = useState({
    face: [], style: null, pose: null,
  });
  const [generationParams, setGenerationParams] = useState({
    prompt: "A photo of a person, cinematic lighting", steps: 30, guidance_scale: 5.0,
    instantid_scale: 0.8, style_shuffle_scale: 0.7, pose_control_scale: 0.75,
    ip_adapter_scale: 0.8, seed: null
  });
  const [generationResult, setGenerationResult] = useState({
    outputImageId: null, imageUrl: null, metadataUrl: null,
    error: null, isLoading: false, statusMessage: '',
  });
  const [canGenerate, setCanGenerate] = useState(false);
  const pollingRef = useRef(null); // To store interval ID

  useEffect(() => {
    setCanGenerate(uploadedFileIds.face.length > 0 && uploadedFileIds.style !== null);
  }, [uploadedFileIds]);

  const handleUploadSuccess = (type, idsOrId) => {
    if (type === 'face') setUploadedFileIds(prev => ({ ...prev, face: idsOrId }));
    else setUploadedFileIds(prev => ({ ...prev, [type]: idsOrId }));
  };

  const handleParamChange = (name, value) => {
    setGenerationParams(prev => ({ ...prev, [name]: value }));
  };

  const stopPolling = () => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  };

  // Effect for polling
  useEffect(() => {
    if (generationResult.isLoading && generationResult.outputImageId && !generationResult.imageUrl && !generationResult.error) {
      stopPolling(); // Clear any existing poll before starting a new one

      pollingRef.current = setInterval(async () => {
        try {
          // 1. Try to fetch metadata
          const metaResponse = await fetch(`${BACKEND_URL}/get_metadata/${generationResult.outputImageId}`);
          if (metaResponse.ok) {
            const metaData = await metaResponse.json();
            if (metaData.status === 'completed') {
              stopPolling();
              setGenerationResult(prev => ({
                ...prev,
                isLoading: false,
                imageUrl: `${BACKEND_URL}/get_image/${prev.outputImageId}`,
                metadataUrl: `${BACKEND_URL}/get_metadata/${prev.outputImageId}`,
                statusMessage: 'Generation complete!',
              }));
            } else if (metaData.status === 'failed') {
              stopPolling();
              setGenerationResult(prev => ({
                ...prev,
                isLoading: false,
                error: metaData.error_message || 'Generation failed (unknown error from metadata).',
                statusMessage: 'Generation failed.',
              }));
            } else {
              // Still processing or other status
              setGenerationResult(prev => ({ ...prev, statusMessage: `Processing... (status: ${metaData.status || 'pending'})` }));
            }
          } else if (metaResponse.status === 404) {
             // Metadata not found yet, keep polling
             setGenerationResult(prev => ({ ...prev, statusMessage: 'Waiting for metadata...' }));
          } else {
            // Other metadata fetch error
            const errorData = await metaResponse.json().catch(() => ({ detail: "Metadata fetch error, non-JSON response" }));
            console.warn("Metadata fetch issue:", metaResponse.status, errorData.detail);
            // Don't stop polling yet, could be a transient server issue
          }
        } catch (err) {
          console.error("Polling error:", err);
          // Potentially stop polling if network error persists, but for now, let it continue a few times
          setGenerationResult(prev => ({ ...prev, statusMessage: `Polling error: ${err.message}. Retrying...` }));
        }
      }, POLLING_INTERVAL);
    }

    // Cleanup function to stop polling when component unmounts or dependencies change
    return () => {
      stopPolling();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generationResult.isLoading, generationResult.outputImageId, generationResult.imageUrl, generationResult.error]);


  const handleGenerate = async () => {
    if (!canGenerate) {
        setGenerationResult(prev => ({ ...prev, error: "Missing face or style images.", statusMessage: "Missing face or style images."}));
        return;
    }
    stopPolling(); // Stop any previous polling before starting a new generation
    setGenerationResult({
        outputImageId: null, imageUrl: null, metadataUrl: null,
        error: null, isLoading: true, statusMessage: 'Sending generation request...'
    });

    const payload = { /* ... (same payload as before, ensure it's correct) ... */
        face_image_ids: uploadedFileIds.face, style_image_id: uploadedFileIds.style,
        pose_image_id: uploadedFileIds.pose, prompt: generationParams.prompt,
        steps: parseInt(generationParams.steps, 10), guidance_scale: parseFloat(generationParams.guidance_scale),
        instantid_scale: parseFloat(generationParams.instantid_scale),
        style_shuffle_scale: parseFloat(generationParams.style_shuffle_scale),
        pose_control_scale: parseFloat(generationParams.pose_control_scale),
        ip_adapter_scale: parseFloat(generationParams.ip_adapter_scale),
        seed: generationParams.seed === null || generationParams.seed === '' ? null : parseInt(generationParams.seed, 10),
    };

    try {
      const response = await fetch(`${BACKEND_URL}/generate_image/`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Generation request failed');

      setGenerationResult(prev => ({ // Keep isLoading true
        ...prev, outputImageId: data.output_image_id, isLoading: true,
        statusMessage: `Task started (ID: ${data.output_image_id}). Waiting for metadata...`,
        error: null, imageUrl: null, metadataUrl: null, // Reset image/meta URLs
      }));
      // Polling will start due to useEffect dependencies
    } catch (error) {
      console.error("Generation trigger error:", error);
      setGenerationResult({
        outputImageId: null, imageUrl: null, metadataUrl: null,
        error: error.message, isLoading: false, statusMessage: `Error: ${error.message}`
      });
    }
  };

  return (
    <div className="App">
      <header className="App-header"><h1>SDXL InstantID Generator</h1></header>
      <main>
        <ImageUploadForm onUploadSuccess={handleUploadSuccess} backendUrl={BACKEND_URL} />
        <GenerationControls
          params={generationParams} onParamChange={handleParamChange}
          onGenerate={handleGenerate} isGenerating={generationResult.isLoading && !generationResult.imageUrl && !generationResult.error}
          canGenerate={canGenerate}
        />
        <OutputDisplay result={generationResult} backendUrl={BACKEND_URL} />
      </main>
    </div>
  );
}
export default App;
