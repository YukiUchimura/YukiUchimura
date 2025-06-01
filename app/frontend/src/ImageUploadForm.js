// app/frontend/src/ImageUploadForm.js
import React, { useState } from 'react';

function ImageUploadForm({ onUploadSuccess, backendUrl }) {
  const [faceFiles, setFaceFiles] = useState([]);
  const [styleFile, setStyleFile] = useState(null);
  const [poseFile, setPoseFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');

  const handleFileChange = (setter) => (event) => {
    if (event.target.files) {
      setter(Array.from(event.target.files)); // Store as array for multi-file, or first for single
    }
  };

  const handleSingleFileChange = (setter) => (event) => {
    if (event.target.files && event.target.files[0]) {
      setter(event.target.files[0]);
    } else {
      setter(null);
    }
  };

  const uploadFiles = async (endpoint, files, typeName, isMultiple = false) => {
    if ((isMultiple && files.length === 0) || (!isMultiple && !files)) {
      setMessage(`Please select ${typeName} image(s).`);
      return;
    }
    setUploading(true);
    setMessage(`Uploading ${typeName}...`);

    const formData = new FormData();
    if (isMultiple) {
      files.forEach(file => formData.append('files', file));
    } else {
      formData.append('file', files); // 'file' for single, 'files' for multiple (matching FastAPI)
    }

    try {
      const response = await fetch(`${backendUrl}${endpoint}`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `Failed to upload ${typeName}`);
      }

      setMessage(`${typeName} uploaded successfully!`);
      if (isMultiple) {
        onUploadSuccess(typeName.toLowerCase(), data.files.map(f => f.id));
      } else {
        onUploadSuccess(typeName.toLowerCase(), data.file.id);
      }
    } catch (error) {
      setMessage(`Error uploading ${typeName}: ${error.message}`);
      console.error(error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-form">
      {message && <p className="message">{message}</p>}

      <div className="form-group">
        <label htmlFor="faceFiles">Face Images (1-5, JPG/PNG):</label>
        <input
          type="file"
          id="faceFiles"
          multiple
          accept="image/jpeg,image/png"
          onChange={handleFileChange(setFaceFiles)}
          disabled={uploading}
        />
        <button onClick={() => uploadFiles('/upload_face_images/', faceFiles, 'face', true)} disabled={uploading || faceFiles.length === 0}>
          Upload Face(s)
        </button>
      </div>

      <div className="form-group">
        <label htmlFor="styleFile">Style Image (JPG/PNG):</label>
        <input
          type="file"
          id="styleFile"
          accept="image/jpeg,image/png"
          onChange={handleSingleFileChange(setStyleFile)}
          disabled={uploading}
        />
        <button onClick={() => uploadFiles('/upload_style_image/', styleFile, 'style')} disabled={uploading || !styleFile}>
          Upload Style
        </button>
      </div>

      <div className="form-group">
        <label htmlFor="poseFile">Pose Image (Optional, PNG):</label>
        <input
          type="file"
          id="poseFile"
          accept="image/png"
          onChange={handleSingleFileChange(setPoseFile)}
          disabled={uploading}
        />
        <button onClick={() => uploadFiles('/upload_pose_image/', poseFile, 'pose')} disabled={uploading || !poseFile}>
          Upload Pose
        </button>
      </div>
    </div>
  );
}

export default ImageUploadForm;
