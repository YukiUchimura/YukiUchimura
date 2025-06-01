// app/frontend/src/GenerationControls.js
import React from 'react';
import './GenerationControls.css'; // Create this CSS file

function GenerationControls({ params, onParamChange, onGenerate, isGenerating, canGenerate }) {
  const handleInputChange = (e) => {
    const { name, value, type } = e.target;
    onParamChange(name, type === 'number' ? parseFloat(value) : value);
  };

  const handleSliderChange = (name, value) => {
    onParamChange(name, parseFloat(value));
  };

  // Helper for slider + number input
  const renderSliderWithInput = (label, name, min, max, step, value) => (
    <div className="param-group">
      <label htmlFor={name}>{label}: {value}</label>
      <input
        type="range"
        id={name}
        name={name}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => handleSliderChange(name, e.target.value)}
        disabled={isGenerating}
      />
      <input
        type="number"
        name={name}
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleInputChange}
        disabled={isGenerating}
        className="param-number-input"
      />
    </div>
  );

  return (
    <div className="generation-controls-container">
      <h3>Generation Parameters</h3>
      <div className="param-group">
        <label htmlFor="prompt">Prompt:</label>
        <textarea
          id="prompt"
          name="prompt"
          value={params.prompt}
          onChange={handleInputChange}
          rows={3}
          disabled={isGenerating}
        />
      </div>

      {renderSliderWithInput("Face Scale (Identity)", "ip_adapter_scale", 0.0, 1.5, 0.05, params.ip_adapter_scale)}
      {/* For simplicity, instantid_scale (ControlNet structure) can use default or be linked if needed */}

      {renderSliderWithInput("Style Scale (Shuffle)", "style_shuffle_scale", 0.0, 1.0, 0.05, params.style_shuffle_scale)}

      {/* Optional: If pose control scale needs UI */}
      {/* renderSliderWithInput("Pose Control Scale", "pose_control_scale", 0.0, 1.0, 0.05, params.pose_control_scale) */}


      <div className="param-group">
        <label htmlFor="steps">Steps:</label>
        <input
          type="number"
          id="steps"
          name="steps"
          min="10"
          max="100"
          step="1"
          value={params.steps}
          onChange={handleInputChange}
          disabled={isGenerating}
        />
      </div>

      <div className="param-group">
        <label htmlFor="seed">Seed (empty for random):</label>
        <input
          type="number"
          id="seed"
          name="seed"
          min="0"
          value={params.seed === null || params.seed === undefined ? '' : params.seed}
          onChange={(e) => onParamChange('seed', e.target.value === '' ? null : parseInt(e.target.value, 10))}
          disabled={isGenerating}
          placeholder="Empty for random"
        />
      </div>

      <button onClick={onGenerate} disabled={isGenerating || !canGenerate} className="generate-button">
        {isGenerating ? 'Generating...' : 'Generate Image'}
      </button>
    </div>
  );
}

export default GenerationControls;
