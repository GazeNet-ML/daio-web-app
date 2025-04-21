"use client";
import { useState } from "react";
import { Upload, FileText } from "lucide-react";
import { CheckCircle } from "lucide-react"; 


export default function UploadVideo() {
    const [file, setFile] = useState(null);
    const [modelOutput, setModelOutput] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const handleFileChange = (event) => {
        const selectedFile = event.target.files[0];
        setFile(selectedFile);
        setModelOutput(null);
        setError(null);
    };

    const handleUpload = async () => {
        if (!file) return;
        setLoading(true);
        setModelOutput(null);
        setError(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("http://localhost:8000/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) throw new Error("Upload failed");
            const data = await response.json();

            console.log("Response from server:", data);

            setModelOutput(data.predictions);
        } catch (error) {
            console.error("Error uploading file:", error);
            setError("Failed to process video. Please try again.");
        }
        setLoading(false);
    };
    
    return (
        
        <div className="min-h-screen bg-gray-100 px-4 py-4">
        <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-teal-600">Gaze Analysis Classifier</h1>
            <p className="text-gray-600 mt-2">
                Upload a video to analyze gaze patterns and classify task engagement and attention characteristics.
           </p>
        </div>

      <div className="container mx-auto max-w-6xl grid grid-cols-1 md:grid-cols-2 gap-6 py-2">
        
        {/* Upload Video Card */}
        <div className="bg-white h-flex border-solid shadow-md rounded-lg p-6">
          <h2 className="text-xl text-black font-semibold mb-4 text-center">Upload Video</h2>
          <div
            className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center 
            hover:border-blue-500 hover:bg-blue-50 transition-all duration-300 
            relative cursor-pointer"
          >
            <input
              type="file"
              accept="video/mp4"
              onChange={handleFileChange}
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
            />
            <div className="flex flex-col items-center justify-center">
              <Upload className="w-12 h-full text-gray-400 mb-4" />
              {file ? (
                <div className="flex items-center">
                  <FileText className="w-5 h-5 mr-2 text-gray-600" />
                  <p className="text-gray-700">{file.name}</p>
                </div>
              ) : (
                <p className="text-gray-500">Drag and drop or click to upload MP4</p>
              )}
            </div>
          </div>

          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="w-full mt-4 px-4 py-2 bg-teal-600 text-white rounded 
            hover:bg-blue-600 transition-colors duration-300
            disabled:bg-teal-600 disabled:cursor-not-allowed"
          >
            {loading ? "Processing..." : "Upload & Process"}
          </button>
        </div>

                {/* Classification Results Card */}
                <div className="bg-white shadow-md rounded-lg p-6 flex flex-col items-center justify-center min-h-[300px] text-center">
  {loading ? (
    <>
      <div className="w-16 h-16 border-4 border-t-transparent border-blue-500 rounded-full animate-spin mb-4" />
      <p className="text-lg text-gray-700 font-medium">Processing video...</p>
    </>
  ) : modelOutput ? (
    <>
      {/* Analysis Completion*/}
      <h2 className="text-xl font-semibold mb-6 text-gray-800">Classification Results</h2>

      <div className="flex justify-center mb-4">
        <div className="bg-green-100 rounded-full p-3">
          <CheckCircle className="w-10 h-10 text-green-500" />
        </div>
      </div>

      <h3 className="text-xl font-semibold text-gray-800 mb-6">Analysis Complete</h3>

      {/* Task Classification */}
      <div className="mb-6 w-full">
        <div className="bg-teal-600 text-white font-medium px-4 py-2 rounded-t-md text-center">
          Task Classification
        </div>
        <div className="border border-t-0 rounded-b-md px-4 py-4 text-center text-xl font-semibold text-gray-800">
          {modelOutput.task.class}
        </div>
      </div>

      {/* Attention Classification */}
      <div className="mb-6 w-full">
        <div className="bg-teal-600 text-white font-medium px-4 py-2 rounded-t-md text-center">
          Attention Classification
        </div>
        <div className="border border-t-0 rounded-b-md px-4 py-4 text-center">
          <p className="text-xl font-semibold text-gray-800">{modelOutput.attention.class}</p>
        </div>
      </div>

      {/* Insights */}
      <div className="bg-teal-50 border border-teal-200 rounded-lg p-6 text-teal-900 shadow-md max-w-4xl mx-auto">
  <h3 className="text-lg font-semibold text-center mb-6">- Analysis Insights -</h3>

  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
    {/* Task Prediction Block */}
    <div className="border border-teal-200 rounded-lg p-4 bg-white shadow-sm space-y-3">
      <h4 className="text-md font-semibold text-center mb-2 text-teal-800">Task Prediction</h4>
      {["Picture", "Reading", "Video"].map((label, idx) => {
        const value = modelOutput.task.probabilities[idx];
        const percentage = (value * 100).toFixed(2);
        return (
          <div key={label} className="flex items-center justify-between gap-4">
            <span className="w-28 text-sm font-medium text-gray-800">{label}</span>
            <div className="relative flex-1 bg-teal-100 rounded h-5">
              <div
                className="absolute left-0 top-0 h-5 bg-teal-400 rounded"
                style={{ width: `${percentage}%` }}
              />
              <span className="absolute right-2 top-0 text-xs font-semibold text-teal-900 leading-5">
                {percentage}%
              </span>
            </div>
          </div>
        );
      })}
    </div>

    {/* Attention Prediction Block */}
    <div className="border border-teal-200 rounded-lg p-4 bg-white shadow-sm space-y-3">
      <h4 className="text-md font-semibold text-center mb-2 text-teal-800">Attention Level Prediction</h4>
      {["BVPS (GTSS)", "BVPS (TSS)", "GVPS (BTSS)", "GVPS (TSS)"].map((label, idx) => {
        const value = modelOutput.attention.probabilities[idx];
        const percentage = (value * 100).toFixed(2);
        return (
          <div key={label} className="flex items-center justify-between gap-4">
            <span className="w-32 text-sm font-medium text-gray-800">{label}</span>
            <div className="relative flex-1 bg-teal-100 rounded h-5">
              <div
                className="absolute left-0 top-0 h-5 bg-teal-400 rounded"
                style={{ width: `${percentage}%` }}
              />
              <span className="absolute right-2 top-0 text-xs font-semibold text-teal-900 leading-5">
                {percentage}%
              </span>
            </div>
          </div>
        );
      })}
    </div>
  </div>
</div>

    </>
  ) : (
    <>
      {/* Placeholder for no result yet */}
      <p className="font-medium text-gray-700">No Results Yet</p>
      <p className="text-sm text-gray-500">Upload a video and run the classifier to see results</p>
    </>
  )}
</div>

            </div>
        </div>
    );
}