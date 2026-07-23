"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Upload, Activity, ShieldAlert, Download, BarChart2 } from 'lucide-react';
import * as ort from 'onnxruntime-web';

// Configure ONNX Runtime to load WASM from CDN to avoid Next.js static serving issues
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/";
ort.env.wasm.numThreads = 1;

type BoundingBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf: number;
};

export default function DentalCariesDetector() {
  const [model, setModel] = useState<ort.InferenceSession | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [detections, setDetections] = useState<BoundingBox[]>([]);
  const [threshold, setThreshold] = useState(0.25);
  const [error, setError] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const originalCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    async function loadModel() {
      try {
        const session = await ort.InferenceSession.create('/best.onnx', { executionProviders: ['wasm'] });
        setModel(session);
        setIsLoading(false);
      } catch (err: any) {
        console.error("Failed to load model:", err);
        setError("Failed to load AI model. Please ensure best.onnx is available in the public folder.");
        setIsLoading(false);
      }
    }
    loadModel();
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setImageSrc(url);
      setDetections([]);
    }
  };

  const processImage = async () => {
    if (!model || !imageSrc || !originalCanvasRef.current || !canvasRef.current) return;
    setIsProcessing(true);
    setDetections([]);

    try {
      const img = new Image();
      img.src = imageSrc;
      await new Promise((resolve) => (img.onload = resolve));

      const inputWidth = 640;
      const inputHeight = 640;

      // Draw original image
      const origCtx = originalCanvasRef.current.getContext('2d');
      if (!origCtx) return;
      originalCanvasRef.current.width = img.width;
      originalCanvasRef.current.height = img.height;
      origCtx.drawImage(img, 0, 0);

      // Preprocess for YOLOv8 (Letterbox padding to 640x640)
      const scale = Math.min(inputWidth / img.width, inputHeight / img.height);
      const scaledW = Math.round(img.width * scale);
      const scaledH = Math.round(img.height * scale);
      const dx = (inputWidth - scaledW) / 2;
      const dy = (inputHeight - scaledH) / 2;

      const preCanvas = document.createElement('canvas');
      preCanvas.width = inputWidth;
      preCanvas.height = inputHeight;
      const preCtx = preCanvas.getContext('2d');
      if (!preCtx) return;
      
      preCtx.fillStyle = '#7a7a7a'; // YOLO typically pads with 114
      preCtx.fillRect(0, 0, inputWidth, inputHeight);
      preCtx.drawImage(img, dx, dy, scaledW, scaledH);

      const imageData = preCtx.getImageData(0, 0, inputWidth, inputHeight);
      const { data } = imageData;

      // Convert to Float32Array [1, 3, 640, 640] and normalize to 0.0 - 1.0
      const float32Data = new Float32Array(1 * 3 * inputWidth * inputHeight);
      for (let i = 0; i < inputWidth * inputHeight; i++) {
        float32Data[i] = data[i * 4] / 255.0; // R
        float32Data[inputWidth * inputHeight + i] = data[i * 4 + 1] / 255.0; // G
        float32Data[2 * inputWidth * inputHeight + i] = data[i * 4 + 2] / 255.0; // B
      }

      const tensor = new ort.Tensor('float32', float32Data, [1, 3, inputHeight, inputWidth]);
      
      const results = await model.run({ images: tensor });
      const output = results['output0'].data as Float32Array;

      // YOLOv8 output: [1, 5, 8400]
      const numElements = 8400;
      let rawDetections: BoundingBox[] = [];

      for (let i = 0; i < numElements; i++) {
        const conf = output[4 * numElements + i];
        if (conf >= threshold) {
          const cx = output[0 * numElements + i];
          const cy = output[1 * numElements + i];
          const w = output[2 * numElements + i];
          const h = output[3 * numElements + i];

          // Reverse letterbox scaling to map back to original image
          let x1 = ((cx - w / 2) - dx) / scale;
          let y1 = ((cy - h / 2) - dy) / scale;
          let x2 = ((cx + w / 2) - dx) / scale;
          let y2 = ((cy + h / 2) - dy) / scale;

          // Clamp
          x1 = Math.max(0, x1);
          y1 = Math.max(0, y1);
          x2 = Math.min(img.width, x2);
          y2 = Math.min(img.height, y2);

          rawDetections.push({ x1, y1, x2, y2, conf });
        }
      }

      // Non-Maximum Suppression (NMS)
      rawDetections.sort((a, b) => b.conf - a.conf);
      const nmsDetections: BoundingBox[] = [];
      const iouThreshold = 0.45;

      for (const det of rawDetections) {
        let keep = true;
        for (const kept of nmsDetections) {
          const areaDet = (det.x2 - det.x1) * (det.y2 - det.y1);
          const areaKept = (kept.x2 - kept.x1) * (kept.y2 - kept.y1);
          const xx1 = Math.max(det.x1, kept.x1);
          const yy1 = Math.max(det.y1, kept.y1);
          const xx2 = Math.min(det.x2, kept.x2);
          const yy2 = Math.min(det.y2, kept.y2);

          const w = Math.max(0, xx2 - xx1);
          const h = Math.max(0, yy2 - yy1);
          const inter = w * h;
          const iou = inter / (areaDet + areaKept - inter);

          if (iou > iouThreshold) {
            keep = false;
            break;
          }
        }
        if (keep) nmsDetections.push(det);
      }

      setDetections(nmsDetections);

      // Draw detections on canvas
      canvasRef.current.width = img.width;
      canvasRef.current.height = img.height;
      const ctx = canvasRef.current.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);

      nmsDetections.forEach((d) => {
        let color = '#ff0000'; // Very Low
        if (d.conf >= 0.8) color = '#00ff00'; // High
        else if (d.conf >= 0.6) color = '#ffff00'; // Medium
        else if (d.conf >= 0.4) color = '#ff8c00'; // Low

        ctx.strokeStyle = color;
        ctx.lineWidth = Math.max(2, img.width / 400);
        ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);

        const text = `Caries ${Math.round(d.conf * 100)}%`;
        ctx.font = `${Math.max(12, img.width / 50)}px Arial`;
        
        const metrics = ctx.measureText(text);
        ctx.fillStyle = color;
        ctx.fillRect(d.x1, d.y1 - 20 - (img.width/100), metrics.width + 10, 20 + (img.width/100));
        
        ctx.fillStyle = '#000';
        ctx.fillText(text, d.x1 + 5, d.y1 - 5);
      });

    } catch (err) {
      console.error(err);
    }
    setIsProcessing(false);
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900" dir="ltr">
      {/* Header */}
      <div className="bg-white border-b border-slate-200">
        <div className="max-w-6xl mx-auto p-6 flex flex-col md:flex-row gap-4 items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-blue-100 text-blue-600 p-3 rounded-xl">
              <ShieldAlert className="w-8 h-8" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-slate-800 tracking-tight">Dental Caries Detection</h1>
              <p className="text-slate-500 font-medium">Browser-side YOLOv8 Inference</p>
            </div>
          </div>
          <div className="text-sm font-semibold text-slate-500 flex items-center gap-2">
            Status: {isLoading ? <span className="text-amber-500 animate-pulse">Loading Model...</span> : <span className="text-emerald-500">Model Ready</span>}
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-8 mt-4">
        {/* Sidebar Controls */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <h2 className="text-lg font-bold mb-4 flex items-center gap-2"><Upload className="w-5 h-5 text-blue-500" /> 1. Upload X-Ray</h2>
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-slate-300 border-dashed rounded-xl cursor-pointer hover:bg-slate-50 transition bg-slate-50/50">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <Upload className="w-8 h-8 text-slate-400 mb-2" />
                <p className="text-sm text-slate-600 font-medium">Click to upload image</p>
              </div>
              <input type="file" className="hidden" accept="image/jpeg, image/png, image/jpg" onChange={handleFileUpload} />
            </label>
          </div>

          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
            <h2 className="text-lg font-bold mb-4 flex items-center gap-2"><Activity className="w-5 h-5 text-emerald-500" /> 2. Configuration</h2>
            <div className="mb-4">
              <label className="block text-sm font-medium text-slate-700 mb-2">
                Confidence Threshold: {Math.round(threshold * 100)}%
              </label>
              <input 
                type="range" 
                min="0" max="100" 
                value={threshold * 100} 
                onChange={(e) => setThreshold(Number(e.target.value) / 100)}
                className="w-full accent-blue-600"
              />
              <p className="text-xs text-slate-500 mt-2">Adjust to filter out low-confidence predictions.</p>
            </div>
            <button
              onClick={processImage}
              disabled={isLoading || isProcessing || !imageSrc}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white font-bold py-3 px-4 rounded-xl transition shadow-sm flex justify-center items-center gap-2"
            >
              {isProcessing ? 'Analyzing...' : 'Analyze Image'}
            </button>
          </div>

          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 text-sm">
            <h3 className="font-bold mb-3">Color Legend</h3>
            <ul className="space-y-2 font-medium">
              <li className="flex items-center gap-2"><div className="w-4 h-4 bg-[#00ff00] rounded"></div> High Confidence (80-100%)</li>
              <li className="flex items-center gap-2"><div className="w-4 h-4 bg-[#ffff00] rounded"></div> Medium Confidence (60-80%)</li>
              <li className="flex items-center gap-2"><div className="w-4 h-4 bg-[#ff8c00] rounded"></div> Low Confidence (40-60%)</li>
              <li className="flex items-center gap-2"><div className="w-4 h-4 bg-[#ff0000] rounded"></div> Very Low Confidence (&lt;40%)</li>
            </ul>
          </div>
        </div>

        {/* Results Area */}
        <div className="lg:col-span-2 space-y-6">
          {error && (
            <div className="bg-red-50 text-red-600 p-4 rounded-xl border border-red-200 font-medium">
              {error}
            </div>
          )}

          <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 flex items-center justify-center min-h-[400px] overflow-hidden">
            {!imageSrc ? (
              <p className="text-slate-400 font-medium text-lg">No image uploaded yet.</p>
            ) : (
              <div className="relative w-full h-full flex items-center justify-center">
                <canvas ref={originalCanvasRef} className="hidden" />
                <canvas 
                  ref={canvasRef} 
                  className="max-w-full max-h-[600px] object-contain rounded-lg shadow-sm border border-slate-200"
                />
                {!isProcessing && detections.length === 0 && imageSrc && !canvasRef.current?.width && (
                   <img src={imageSrc} className="max-w-full max-h-[600px] object-contain rounded-lg" alt="preview" />
                )}
              </div>
            )}
          </div>

          {detections.length > 0 && (
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-xl font-bold mb-4 flex items-center gap-2"><BarChart2 className="w-5 h-5 text-blue-500" /> Detection Results</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center">
                  <div className="text-3xl font-bold text-slate-800">{detections.length}</div>
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Caries Found</div>
                </div>
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 text-center">
                  <div className="text-3xl font-bold text-slate-800">{Math.round(detections.reduce((a, b) => a + b.conf, 0) / detections.length * 100)}%</div>
                  <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Avg Confidence</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
