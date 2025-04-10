"use client";
import { useState } from "react";
import { Upload, FileText } from "lucide-react";

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

    const renderProbabilityBar = (probabilities, labels) => {
        return probabilities.map((prob, index) => (
            <div key={index} className="flex items-center mb-2">
                <span className="w-40 mr-2">{labels[index]}</span>
                <div 
                    className="h-5 text-right px-1 text-black" 
                    style={{
                        width: `${prob * 100}%`, 
                        backgroundColor: `rgba(33, 150, 243, ${prob})`
                    }}
                >
                    {(prob * 100).toFixed(2)}%
                </div>
            </div>
        ));
    };

    return (
        <div className="container mx-auto max-w-md p-5">
            <div className="bg-white shadow-md rounded-lg p-6">
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
                        <Upload className="w-12 h-12 text-gray-400 mb-4" />
                        {file ? (
                            <div className="flex items-center">
                                <FileText className="w-5 h-5 mr-2 text-gray-600" />
                                <p className="text-gray-700">{file.name}</p>
                            </div>
                        ) : (
                            <p className="text-gray-500">
                                Drag and drop or click to upload MP4
                            </p>
                        )}
                    </div>
                </div>

                <button 
                    onClick={handleUpload} 
                    disabled={!file || loading}
                    className="w-full mt-4 px-4 py-2 bg-blue-500 text-white rounded 
                    hover:bg-blue-600 transition-colors duration-300
                    disabled:bg-gray-400 disabled:cursor-not-allowed"
                >
                    {loading ? "Processing..." : "Upload & Process"}
                </button>

                {error && <p className="text-red-500 mt-3 text-center">{error}</p>}

                {modelOutput && (
                    <div className="mt-5 text-black">
                        <h3 className="text-lg font-semibold mb-3">Model Predictions</h3>
                        
                        <div className="mb-4">
                            <h4 className="font-medium mb-2">Task Prediction</h4>
                            <p>Predicted Task: <strong>{modelOutput.task.class}</strong></p>
                            <div className="mt-2">
                                {renderProbabilityBar(
                                    modelOutput.task.probabilities, 
                                    ["Picture", "Reading", "Video"]
                                )}
                            </div>
                        </div>

                        <div>
                            <h4 className="font-medium mb-2">Attention Level Prediction</h4>
                            <p>Predicted Attention Level: <strong>{modelOutput.attention.class}</strong></p>
                            <div className="mt-2">
                                {renderProbabilityBar(
                                    modelOutput.attention.probabilities, 
                                    ["BVPS (GTSS)", "BVPS (TSS)", "GVPS (BTSS)", "GVPS (TSS)"]
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}