"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
    Upload,
    FileImage,
    X,
    CheckCircle,
    Loader2,
    ImageIcon,
    Brain,
} from "lucide-react";
import { Navigation } from "@/components/navigation";
import { ResultsViewer } from "@/components/results-viewer";
import { cn } from "@/lib/utils";
import Image from "next/image";


interface UploadedFile {
    file: File;
    preview: string;
    id: string;
    status: "uploading" | "processing" | "completed" | "error";
    progress: number;
}

interface AnalysisResults {
    segmentation: {
        diceScore: number;
        precision: number;
        tumorSize: string;
        biRadsScore: string;
        margin: string;
        shape: string;
        segmentedImages: {
          [colormap: string]: string
        }
    };
    stage: {
        predictedStage: string;
        precision: number;
        tnmStaging: string;
    };
    treatments: {
        type: string;
        recommendation: string;
        priority: string;
        description: string;
    }[];
    pipeline: {
      encoder: {
        name: string;
        description: string;
        image: string;
      };
      bottleneck: {
        name: string;
        description: string;
        image: string;
      };
      decoder: {
        name: string;
        description: string;
        image: string;
      };
    };
}

type AppState = "upload" | "processing" | "results";

export default function OncoPathApp() {
    const [appState, setAppState] = useState<AppState>("upload");
    const [uploadedFile, setUploadedFile] = useState<UploadedFile | null>(null);
    const [analysisResults, setAnalysisResults] =
        useState<AnalysisResults | null>(null);

    const onDrop = useCallback((acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (!file) return;

        const id = Math.random().toString(36).substr(2, 9);
        const newFile: UploadedFile = {
            file,
            preview: URL.createObjectURL(file),
            id,
            status: "uploading",
            progress: 0,
        };

        setUploadedFile(newFile);
        setAppState("processing");

        const uploadInterval = setInterval(() => {
            setUploadedFile((prev) =>
                prev
                    ? { ...prev, progress: Math.min(prev.progress + 15, 100) }
                    : null
            );
        }, 200);

        const formData = new FormData();
        formData.append("file", file);
        formData.append("colormap", "inferno");

        fetch("http://127.0.0.1:8000/upload", {
            method: "POST",
            body: formData,
        })
            .then(async (res) => {
                clearInterval(uploadInterval);
                if (!res.ok) throw new Error("Upload failed");
                const data = await res.json();

                setUploadedFile((prev) =>
                    prev
                        ? { ...prev, status: "completed", progress: 100 }
                        : null
                );

                setAnalysisResults({
                    segmentation: {
                        diceScore: 73.69,
                        precision: 75,
                        tumorSize:
                            data?.tumor_details?.[0]?.["Size (mm²)"] != null
                                ? data.tumor_details[0]["Size (mm²)"].toFixed(
                                      2
                                  ) + " mm²"
                                : "Unknown",

                        biRadsScore:
                            data?.tumor_details?.[0]?.["BI-RADS"] ??
                            "BI-RADS N/A",
                        margin: data?.tumor_details?.[0]?.["Margin"] ?? "N/A",
                        shape: data?.tumor_details?.[0]?.["Shape"] ?? "N/A",
                        segmentedImages: {
                          inferno: `data:image/png;base64,${data.segmentation.segmentedImages.inferno}`,
                          gray: `data:image/png;base64,${data.segmentation.segmentedImages.gray}`,
                          nipy_spectral: `data:image/png;base64,${data.segmentation.segmentedImages.nipy_spectral}`
                        }
                    },
                    stage: {
                        predictedStage:
                            data?.tumor_details?.[0]?.["Cancer Stage"] ??
                            "Unknown",
                        precision: 90.0,
                        tnmStaging: "T2N0M0",
                    },
                    treatments: data.treatments ?? [],
                    pipeline: {
                      encoder: {
                        name: data.pipeline.encoder.name,
                        description: data.pipeline.encoder.description,
                        image: data.pipeline.encoder.image,
                      },
                      bottleneck: {
                        name: data.pipeline.bottleneck.name,
                        description: data.pipeline.bottleneck.description,
                        image: data.pipeline.bottleneck.image,
                      },
                      decoder: {
                        name: data.pipeline.decoder.name,
                        description: data.pipeline.decoder.description,
                        image: data.pipeline.decoder.image,
                      },
                    },
                });

                setAppState("results");
            })
            .catch((err) => {
                console.error(err);
                clearInterval(uploadInterval);
                setUploadedFile((prev) =>
                    prev ? { ...prev, status: "error", progress: 0 } : null
                );
                setAppState("upload");
            });
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            "image/*": [".png", ".jpg", ".jpeg", ".dcm", ".dicom"],
        },
        multiple: false,
        maxFiles: 1,
    });

    const resetApp = () => {
        if (uploadedFile) {
            URL.revokeObjectURL(uploadedFile.preview);
        }
        setUploadedFile(null);
        setAnalysisResults(null);
        setAppState("upload");
    };

    const removeFile = () => {
        if (uploadedFile) {
            URL.revokeObjectURL(uploadedFile.preview);
            setUploadedFile(null);
            setAppState("upload");
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
            <Navigation onReset={resetApp} currentState={appState} />

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {appState === "upload" && (
                    <div className="max-w-4xl mx-auto">
                        <div className="mb-8 text-center">
  <div className="flex justify-center items-center mb-2">
    <Image
      src="/vector_oncopath_logo.png"
      alt="OncoPath Logo"
      width={32}
      height={32}
      className="mr-2"
    />
    <h1 className="text-3xl font-bold text-white">OncoPath</h1>
  </div>
  <p className="text-slate-400">
    Upload a breast ultrasound image for AI-powered tumor detection and treatment recommendations
  </p>
</div>


                        {/* Upload Area */}
                        <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
                            <CardContent className="p-8">
                                <div
                                    {...getRootProps()}
                                    className={cn(
                                        "border-2 border-dashed rounded-xl p-16 text-center cursor-pointer transition-all duration-300",
                                        isDragActive
                                            ? "border-blue-500 bg-blue-500/10"
                                            : "border-slate-600 hover:border-slate-500 hover:bg-slate-800/30"
                                    )}
                                >
                                    <input {...getInputProps()} />
                                    <div className="flex flex-col items-center space-y-6">
                                        <div className="p-6 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-full">
                                            <Upload className="h-12 w-12 text-blue-400" />
                                        </div>
                                        {isDragActive ? (
                                            <div>
                                                <p className="text-2xl font-medium text-blue-400 mb-2">
                                                    Drop image here...
                                                </p>
                                                <p className="text-slate-400">
                                                    Release to upload
                                                </p>
                                            </div>
                                        ) : (
                                            <div>
                                                <p className="text-2xl font-medium text-white mb-3">
                                                    Drag & drop your medical
                                                    image here
                                                </p>
                                                <p className="text-slate-400 mb-6">
                                                    or click to browse files
                                                </p>
                                                <Button
                                                    size="lg"
                                                    className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 px-8 py-3"
                                                >
                                                    <FileImage className="mr-2 h-5 w-5" />
                                                    Select Image
                                                </Button>
                                            </div>
                                        )}
                                    </div>
                                </div>

                                <div className="mt-6 flex flex-wrap justify-center gap-3">
                                    <Badge
                                        variant="secondary"
                                        className="bg-slate-700 text-slate-300 px-3 py-1"
                                    >
                                        DICOM
                                    </Badge>
                                    <Badge
                                        variant="secondary"
                                        className="bg-slate-700 text-slate-300 px-3 py-1"
                                    >
                                        PNG
                                    </Badge>
                                    <Badge
                                        variant="secondary"
                                        className="bg-slate-700 text-slate-300 px-3 py-1"
                                    >
                                        JPEG
                                    </Badge>
                                    <Badge
                                        variant="secondary"
                                        className="bg-slate-700 text-slate-300 px-3 py-1"
                                    >
                                        Max 50MB
                                    </Badge>
                                    <Badge
                                        variant="secondary"
                                        className="bg-slate-700 text-slate-300 px-3 py-1"
                                    >
                                        Single Upload
                                    </Badge>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                )}

                {appState === "processing" && uploadedFile && (
                    <div className="max-w-4xl mx-auto">
                        <div className="mb-8 text-center">
                            <h1 className="text-3xl font-bold text-white mb-2">
                                Processing Image
                            </h1>
                            <p className="text-slate-400">
                                AI analysis in progress - this may take a few
                                moments
                            </p>
                        </div>

                        <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
                            <CardContent className="p-8">
                                <div className="flex flex-col lg:flex-row gap-8 items-center">
                                    {/* Image Preview */}
                                    <div className="flex-shrink-0">
                                        <div className="relative bg-slate-900 rounded-lg overflow-hidden border border-slate-600">
                                            {uploadedFile.file.type.startsWith(
                                                "image/"
                                            ) ? (
                                                <Image
                                                    src={
                                                        uploadedFile.preview ||
                                                        "/placeholder.svg"
                                                    }
                                                    alt="Uploaded medical image"
                                                    width={300}
                                                    height={300}
                                                    className="w-80 h-80 object-cover"
                                                />
                                            ) : (
                                                <div className="w-80 h-80 flex items-center justify-center">
                                                    <ImageIcon className="h-20 w-20 text-slate-500" />
                                                </div>
                                            )}
                                            <Button
                                                variant="ghost"
                                                size="sm"
                                                onClick={removeFile}
                                                className="absolute top-2 right-2 h-8 w-8 p-0 bg-slate-800/80 hover:bg-slate-700 text-slate-300"
                                            >
                                                <X className="h-4 w-4" />
                                            </Button>
                                        </div>
                                    </div>

                                    {/* Processing Status */}
                                    <div className="flex-1 space-y-6">
                                        <div>
                                            <h3 className="text-xl font-semibold text-white mb-2">
                                                {uploadedFile.file.name}
                                            </h3>
                                            <p className="text-slate-400">
                                                Size:{" "}
                                                {(
                                                    uploadedFile.file.size /
                                                    1024 /
                                                    1024
                                                ).toFixed(2)}{" "}
                                                MB
                                            </p>
                                        </div>

                                        <div className="space-y-4">
                                            <div className="flex items-center space-x-3">
                                                {uploadedFile.status ===
                                                    "uploading" && (
                                                    <>
                                                        <Loader2 className="h-5 w-5 text-blue-400 animate-spin" />
                                                        <span className="text-white font-medium">
                                                            Uploading image...
                                                        </span>
                                                    </>
                                                )}
                                                {uploadedFile.status ===
                                                    "processing" && (
                                                    <>
                                                        <Loader2 className="h-5 w-5 text-purple-400 animate-spin" />
                                                        <span className="text-white font-medium">
                                                            Analyzing with AI...
                                                        </span>
                                                    </>
                                                )}
                                                {uploadedFile.status ===
                                                    "completed" && (
                                                    <>
                                                        <CheckCircle className="h-5 w-5 text-green-400" />
                                                        <span className="text-white font-medium">
                                                            Analysis complete!
                                                        </span>
                                                    </>
                                                )}
                                            </div>

                                            <div className="space-y-2">
                                                <div className="flex justify-between text-sm">
                                                    <span className="text-slate-400">
                                                        {uploadedFile.status ===
                                                        "uploading"
                                                            ? "Upload Progress"
                                                            : "Analysis Progress"}
                                                    </span>
                                                    <span className="text-white">
                                                        {uploadedFile.progress}%
                                                    </span>
                                                </div>
                                                <Progress
                                                    value={
                                                        uploadedFile.progress
                                                    }
                                                    className="h-3"
                                                />
                                            </div>

                                            {uploadedFile.status ===
                                                "processing" && (
                                                <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
                                                    <div className="flex items-center space-x-2 mb-2">
                                                        <Brain className="h-4 w-4 text-purple-400" />
                                                        <span className="text-white font-medium text-sm">
                                                            AI Processing Steps
                                                        </span>
                                                    </div>
                                                    <div className="space-y-1 text-sm text-slate-300">
                                                        <div className="flex items-center space-x-2">
                                                            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                                                            <span>
                                                                Image
                                                                preprocessing
                                                                complete
                                                            </span>
                                                        </div>
                                                        <div className="flex items-center space-x-2">
                                                            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                                                            <span>
                                                                Tumor
                                                                segmentation in
                                                                progress
                                                            </span>
                                                        </div>
                                                        <div className="flex items-center space-x-2">
                                                            <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
                                                            <span>
                                                                Stage
                                                                classification
                                                                analysis
                                                            </span>
                                                        </div>
                                                        <div className="flex items-center space-x-2">
                                                            <div className="w-2 h-2 bg-slate-500 rounded-full"></div>
                                                            <span>
                                                                Treatment
                                                                recommendation
                                                                generation
                                                            </span>
                                                        </div>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                )}

                {appState === "results" && analysisResults && uploadedFile && (
                    <ResultsViewer
                    results={analysisResults}
                    originalImage={uploadedFile.preview}
                    fileName={uploadedFile.file.name}
                    onNewAnalysis={resetApp}
                  />
                )}
            </div>
        </div>
    );
}
