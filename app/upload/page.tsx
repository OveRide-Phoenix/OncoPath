"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Upload, FileImage, X, CheckCircle, AlertCircle, Loader2, ImageIcon } from "lucide-react"
import { Navigation } from "@/components/navigation"
import { cn } from "@/lib/utils"
import Image from "next/image"

interface UploadedFile {
  file: File
  preview: string
  id: string
  status: "uploading" | "processing" | "completed" | "error"
  progress: number
}

export default function UploadPage() {
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [isProcessing, setIsProcessing] = useState(false)

  const onDrop = useCallback((acceptedFiles: File[]) => {
    acceptedFiles.forEach(async (file) => {
      const id = Math.random().toString(36).substr(2, 9);
      const newFile: UploadedFile = {
        file,
        preview: URL.createObjectURL(file),
        id,
        status: "uploading",
        progress: 0,
      };
  
      setFiles((prev) => [...prev, newFile]);
  
      const formData = new FormData();
      formData.append("file", file);
      formData.append("colormap", "inferno");
  
      try {
        const res = await fetch("http://127.0.0.1:5000/upload", {
          method: "POST",
          body: formData,
        });
  
        if (!res.ok) throw new Error("Upload failed");
  
        const data = await res.json();
        console.log("Prediction result:", data);
  
        setFiles((prev) =>
          prev.map((f) =>
            f.id === id ? { ...f, status: "completed", progress: 100 } : f,
          ),
        );
  
        // Optionally route to result page and pass data via state or useContext
      } catch (err) {
        console.error("Upload failed:", err);
        setFiles((prev) =>
          prev.map((f) =>
            f.id === id ? { ...f, status: "error", progress: 0 } : f,
          ),
        );
      }
    });
  }, []);  

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".png", ".jpg", ".jpeg", ".dcm", ".dicom"],
    },
    multiple: true,
  })

  const removeFile = (id: string) => {
    setFiles((prev) => {
      const fileToRemove = prev.find((f) => f.id === id)
      if (fileToRemove) {
        URL.revokeObjectURL(fileToRemove.preview)
      }
      return prev.filter((f) => f.id !== id)
    })
  }

  const processAll = () => {
    setIsProcessing(true)
    setTimeout(() => {
      setIsProcessing(false)
      // Redirect to results page
      window.location.href = "/results"
    }, 3000)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <Navigation onReset={function (): void {
        throw new Error("Function not implemented.")
      } } currentState={"processing"} />

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Upload Medical Images</h1>
          <p className="text-slate-400">
            Upload breast ultrasound images in DICOM, PNG, or JPEG format for AI analysis
          </p>
        </div>

        {/* Upload Area */}
        <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm mb-8">
          <CardContent className="p-8">
            <div
              {...getRootProps()}
              className={cn(
                "border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-300",
                isDragActive
                  ? "border-blue-500 bg-blue-500/10"
                  : "border-slate-600 hover:border-slate-500 hover:bg-slate-800/30",
              )}
            >
              <input {...getInputProps()} />
              <div className="flex flex-col items-center space-y-4">
                <div className="p-4 bg-slate-700/50 rounded-full">
                  <Upload className="h-8 w-8 text-slate-300" />
                </div>
                {isDragActive ? (
                  <div>
                    <p className="text-lg font-medium text-blue-400">Drop files here...</p>
                    <p className="text-sm text-slate-400">Release to upload</p>
                  </div>
                ) : (
                  <div>
                    <p className="text-lg font-medium text-white mb-2">Drag & drop medical images here</p>
                    <p className="text-sm text-slate-400 mb-4">or click to browse files</p>
                    <Button className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
                      <FileImage className="mr-2 h-4 w-4" />
                      Select Files
                    </Button>
                  </div>
                )}
              </div>
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              <Badge variant="secondary" className="bg-slate-700 text-slate-300">
                DICOM
              </Badge>
              <Badge variant="secondary" className="bg-slate-700 text-slate-300">
                PNG
              </Badge>
              <Badge variant="secondary" className="bg-slate-700 text-slate-300">
                JPEG
              </Badge>
              <Badge variant="secondary" className="bg-slate-700 text-slate-300">
                Max 50MB per file
              </Badge>
            </div>
          </CardContent>
        </Card>

        {/* File List */}
        {files.length > 0 && (
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-white">Uploaded Files ({files.length})</CardTitle>
              {files.some((f) => f.status === "completed") && (
                <Button
                  onClick={processAll}
                  disabled={isProcessing}
                  className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <CheckCircle className="mr-2 h-4 w-4" />
                      Analyze All
                    </>
                  )}
                </Button>
              )}
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {files.map((uploadedFile) => (
                  <div
                    key={uploadedFile.id}
                    className="relative bg-slate-700/50 rounded-lg p-4 border border-slate-600"
                  >
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeFile(uploadedFile.id)}
                      className="absolute top-2 right-2 h-6 w-6 p-0 text-slate-400 hover:text-white"
                    >
                      <X className="h-4 w-4" />
                    </Button>

                    <div className="aspect-square bg-slate-800 rounded-lg mb-3 overflow-hidden">
                      {uploadedFile.file.type.startsWith("image/") ? (
                        <Image
                          src={uploadedFile.preview || "/placeholder.svg"}
                          alt="Preview"
                          width={200}
                          height={200}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <ImageIcon className="h-12 w-12 text-slate-500" />
                        </div>
                      )}
                    </div>

                    <div className="space-y-2">
                      <p className="text-sm font-medium text-white truncate">{uploadedFile.file.name}</p>
                      <p className="text-xs text-slate-400">{(uploadedFile.file.size / 1024 / 1024).toFixed(2)} MB</p>

                      <div className="flex items-center space-x-2">
                        {uploadedFile.status === "uploading" && (
                          <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />
                        )}
                        {uploadedFile.status === "processing" && (
                          <Loader2 className="h-4 w-4 text-yellow-400 animate-spin" />
                        )}
                        {uploadedFile.status === "completed" && <CheckCircle className="h-4 w-4 text-green-400" />}
                        {uploadedFile.status === "error" && <AlertCircle className="h-4 w-4 text-red-400" />}

                        <Badge
                          variant="secondary"
                          className={cn(
                            "text-xs",
                            uploadedFile.status === "uploading" && "bg-blue-500/20 text-blue-300",
                            uploadedFile.status === "processing" && "bg-yellow-500/20 text-yellow-300",
                            uploadedFile.status === "completed" && "bg-green-500/20 text-green-300",
                            uploadedFile.status === "error" && "bg-red-500/20 text-red-300",
                          )}
                        >
                          {uploadedFile.status === "uploading" && "Uploading"}
                          {uploadedFile.status === "processing" && "Processing"}
                          {uploadedFile.status === "completed" && "Ready"}
                          {uploadedFile.status === "error" && "Error"}
                        </Badge>
                      </div>

                      <Progress value={uploadedFile.progress} className="h-2" />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
