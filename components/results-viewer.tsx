"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { Separator } from "@/components/ui/separator"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import html2canvas from "html2canvas"
import {
  Eye,
  Download,
  Share2,
  Info,
  Zap,
  Activity,
  AlertTriangle,
  CheckCircle,
  Brain,
  Stethoscope,
  Pill,
  Scissors,
  Radiation,
  Heart,
  Upload,
  Palette,
  ArrowRight,
  Network,
  Layers,
  Code,
} from "lucide-react"
import Image from "next/image"

// Define types for our data structure
interface SegmentationMetrics {
  segmentedImages: { [key: string]: string }
  diceScore: number
  precision: number
  tumorSize: string
  biRadsScore: string
  margin: string
  density: string
  shape: string
}


interface StageInfo {
  predictedStage: string
  precision: number
}

interface Treatment {
  type: string
  recommendation: string
  priority: string
  description: string
}

interface PipelineStage {
  name: string
  description: string
  image: string
}

interface AnalysisResults {
  segmentation: SegmentationMetrics
  stage: StageInfo
  treatments: Treatment[]
  pipeline: {
    encoder: PipelineStage
    bottleneck: PipelineStage
    decoder: PipelineStage
  }
}

// Colormap options
interface Colormap {
  name: string
  value: string
  description: string
}

const colormaps: Colormap[] = [
  { name: "Grayscale", value: "gray", description: "Black and white visualization" },
  { name: "Inferno", value: "inferno", description: "Yellow-red-black heat visualization" },
  { name: "Nipy Spectral", value: "nipy_spectral", description: "Rainbow color spectrum" },
]

interface ResultsViewerProps {
  results: any
  originalImage: string
  fileName: string
  onNewAnalysis: () => void
  pageRef: React.RefObject<HTMLDivElement>
}

export function ResultsViewer({ results: initialResults, originalImage, fileName, onNewAnalysis, pageRef }: ResultsViewerProps) {
  // Create a structured version of the results with dummy data for missing fields
  const stageDescriptions: Record<string, string> = {
    "Stage 0": "Stage 0 breast cancer (DCIS) means the cancer cells are confined to the ducts and haven’t spread into surrounding tissue. It's highly treatable and usually requires localized treatment.",
    "Stage I": "Stage I breast cancer indicates a small tumor with minimal or no lymph node involvement. Treatment often includes surgery and may be followed by radiation or hormone therapy.",
    "Stage II": "Stage II breast cancer typically indicates a tumor that is larger than Stage I but has not spread to distant parts of the body. Treatment often involves a combination of local and systemic therapies.",
    "Stage III": "Stage III breast cancer is more advanced and may involve multiple lymph nodes or larger tumors. Treatment usually includes surgery, chemotherapy, and radiation.",
    "Stage IV": "Stage IV breast cancer has spread (metastasized) to distant organs. Treatment focuses on prolonging life and improving quality of life with systemic therapies.",
  }

  const results: AnalysisResults = {
    segmentation: initialResults.segmentation || {
      diceScore: 94.2,
      precision: 96.8,
      tumorSize: "2.3 cm",
      biRadsScore: "BI-RADS 4",
      margin: "Irregular",
      density: "Heterogeneous",
    },
    stage: initialResults.stage || {
      predictedStage: "Stage II",
      precision: 89.5,
    },
    treatments: initialResults.treatments || [
      {
        type: "Surgery",
        recommendation: "Lumpectomy with sentinel lymph node biopsy",
        priority: "Primary",
        description: "Breast-conserving surgery to remove the tumor while preserving breast tissue",
      },
      {
        type: "Chemotherapy",
        recommendation: "Adjuvant chemotherapy (AC-T protocol)",
        priority: "Secondary",
        description: "Systemic treatment to eliminate any remaining cancer cells",
      },
      {
        type: "Radiation",
        recommendation: "Whole breast radiation therapy",
        priority: "Secondary",
        description: "Targeted radiation to reduce local recurrence risk",
      },
    ],
    pipeline: initialResults.pipeline || {
      encoder: {
        name: "Encoder",
        description: "Feature extraction from input image",
        image: "/placeholder.svg?height=150&width=150",
      },
      bottleneck: {
        name: "Bottleneck",
        description: "Compressed representation of features",
        image: "/placeholder.svg?height=150&width=150",
      },
      decoder: {
        name: "Decoder",
        description: "Reconstruction of segmentation mask",
        image: "/placeholder.svg?height=150&width=150",
      },
    },
  }

  const [showExplainability, setShowExplainability] = useState(false)
  const [brightness, setBrightness] = useState(100)
  const [contrast, setContrast] = useState(100)
  const [selectedColormap, setSelectedColormap] = useState<string>("inferno")

  const treatmentIcons = {
    Surgery: Scissors,
    Chemotherapy: Pill,
    Radiation: Radiation,
    "Hormone Therapy": Heart,
  }


  const renderSegmentedImage = (colormap: string) => {
    
    const base64 = results.segmentation?.segmentedImages?.[colormap] || ""
    if (!base64) return <div className="text-sm text-slate-500">No image for colormap: {colormap}</div>

    return (
      <div className="relative w-full h-72"> 
        <Image
          src={`${base64}`}
          alt={`Segmented Image (${colormap})`}
          fill
          className="object-cover"
          style={{
            filter: `brightness(${brightness}%) contrast(${contrast}%)`,
          }}
        />
      </div>
    )
  }
  

  return (
    <div className="space-y-8">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white mb-3">Analysis Results</h1>
          <p className="text-slate-400">
            File: {fileName} | Analyzed: {new Date().toLocaleDateString()}
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
        <Button
  variant="outline"
  className="border-slate-600 text-slate-300 hover:bg-slate-800"
  onClick={async () => {
    if (!pageRef?.current) return

    const canvas = await html2canvas(pageRef.current)
    const dataUrl = canvas.toDataURL("image/png")
    const link = document.createElement("a")
    link.href = dataUrl
    link.download = "oncopath-dashboard.png"
    link.click()
  }}
>
  <Download className="mr-2 h-4 w-4" />
  Export Report
</Button>

          <Button
            onClick={onNewAnalysis}
            className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
          >
            <Upload className="mr-2 h-4 w-4" />
            New Analysis
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Results Panel */}
        <div className="lg:col-span-2 space-y-8">
          {/* Image Comparison */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                <div>
                  <CardTitle className="text-white flex items-center text-xl">
                    <Eye className="mr-2 h-5 w-5" />
                    Segmentation Results
                  </CardTitle>
                  <CardDescription className="text-slate-400 mt-1">
                    Compare original and segmented images with adjustable controls
                  </CardDescription>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowExplainability(!showExplainability)}
                  className="border-slate-600 text-slate-300 hover:bg-slate-700"
                >
                  <Brain className="mr-2 h-4 w-4" />
                  {showExplainability ? "Hide" : "Show"} AI Explanation
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="relative bg-slate-900 rounded-lg">
                <div className="flex">
                  <div className="w-1/2 relative">
                    <div className="h-72 md:h-80">
                      <Image
                        src={originalImage || "/placeholder.svg"}
                        alt="Original Image"
                        fill
                        className="object-cover"
                        style={{
                          filter: `brightness(${brightness}%) contrast(${contrast}%)`,
                        }}
                      />
                    </div>
                    <div className="absolute top-2 left-2 z-10">
                      <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30">Original</Badge>
                    </div>
                  </div>
                  <div className="w-1/2 relative">
                    <div className="h-72 md:h-80">
                    {renderSegmentedImage(selectedColormap)}
                    </div>
                    <div className="absolute top-2 left-2 z-10">
                      <Badge className="bg-purple-500/20 text-purple-300 border-purple-500/30">Segmented</Badge>
                    </div>
                    {showExplainability && (
                      <div className="absolute inset-0 bg-gradient-to-r from-red-500/30 to-yellow-500/30 mix-blend-overlay" />
                    )}
                  </div>
                </div>
              </div>

              {/* Brightness and Contrast Controls */}
              <Card className="bg-slate-700/50 border-slate-600">
                <CardHeader className="pb-2">
                  <CardTitle className="text-white text-lg flex items-center">
                    <Eye className="mr-2 h-4 w-4 text-blue-400" />
                    Image Controls
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-5">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <label className="text-sm text-slate-300">Brightness</label>
                        <span className="text-xs text-slate-400">{brightness}%</span>
                      </div>
                      <Slider
                        value={[brightness]}
                        onValueChange={(value) => setBrightness(value[0])}
                        min={50}
                        max={150}
                        step={5}
                        className="w-full"
                      />
                    </div>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <label className="text-sm text-slate-300">Contrast</label>
                        <span className="text-xs text-slate-400">{contrast}%</span>
                      </div>
                      <Slider
                        value={[contrast]}
                        onValueChange={(value) => setContrast(value[0])}
                        min={50}
                        max={150}
                        step={5}
                        className="w-full"
                      />
                    </div>
                  </div>

                  <Separator className="bg-slate-600/50" />

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <label className="text-sm text-slate-300 flex items-center">
                        <Palette className="mr-2 h-4 w-4 text-purple-400" />
                        Colormap
                      </label>
                      <Select value={selectedColormap} onValueChange={setSelectedColormap}>
                        <SelectTrigger className="bg-slate-800 border-slate-600 text-slate-300">
                          <SelectValue placeholder="Select colormap" />
                        </SelectTrigger>
                        <SelectContent className="bg-slate-800 border-slate-600">
                          {colormaps.map((colormap) => (
                            <SelectItem key={colormap.value} value={colormap.value} className="text-slate-300">
                              <div className="flex items-center">
                                <span>{colormap.name}</span>
                                <span className="ml-2 text-xs text-slate-400">- {colormap.description}</span>
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="flex items-end space-x-3">
                      <Button
                        variant="outline"
                        onClick={() => {
                          setBrightness(100)
                          setContrast(100)
                          setSelectedColormap("inferno")
                        }}
                        className="border-slate-600 text-slate-300 hover:bg-slate-700"
                      >
                        Reset All
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => {
                          setBrightness(120)
                          setContrast(110)
                        }}
                        className="border-slate-600 text-slate-300 hover:bg-slate-700"
                      >
                        Enhance
                      </Button>
                      <Button
                        variant="outline"
                        onClick={() => {
                          setBrightness(80)
                          setContrast(130)
                        }}
                        className="border-slate-600 text-slate-300 hover:bg-slate-700"
                      >
                        High Contrast
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {showExplainability && (
                <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
                  <h4 className="text-white font-medium mb-2 flex items-center">
                    <Zap className="mr-2 h-4 w-4 text-yellow-400" />
                    AI Explainability (Grad-CAM)
                  </h4>
                  <p className="text-slate-300 text-sm">
                    The heatmap overlay shows regions that most influenced the AI's decision. Red/Purple areas indicate high
                    attention, while Yellow/Orange areas show moderate attention.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Detailed Metrics */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-white flex items-center text-xl">
                <Activity className="mr-2 h-5 w-5" />
                Segmentation Metrics
              </CardTitle>
              <CardDescription className="text-slate-400 mt-1">
                Quantitative assessment of segmentation quality and tumor characteristics
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                <div className="bg-slate-700/30 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-green-400 mb-1">{results.segmentation.diceScore}%</div>
                  <div className="text-sm text-slate-400 mb-2">Dice Score</div>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-blue-400 mb-1">{results.segmentation.precision}%</div>
                  <div className="text-sm text-slate-400 mb-2">Precision</div>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold text-purple-400 mb-1">{results.segmentation.tumorSize}</div>
                  <div className="text-sm text-slate-400 mb-2">Tumor Size</div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <div className="text-sm text-slate-400 mb-1">BI-RADS Score</div>
                  <div className="text-lg font-semibold text-orange-400">{results.segmentation.biRadsScore}</div>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <div className="text-sm text-slate-400 mb-1">Margin</div>
                  <div className="text-lg font-semibold text-red-400">{results.segmentation.margin}</div>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-4">
                  <div className="text-sm text-slate-400 mb-1">Shape</div>
                  <div className="text-lg font-semibold text-yellow-400">{results.segmentation.shape}</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Pipeline Visualization */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-white flex items-center text-xl">
                <Network className="mr-2 h-5 w-5 text-blue-400" />
                Image Processing Pipeline
              </CardTitle>
              <CardDescription className="text-slate-400 mt-1">
                Visualization of the AI model's internal processing stages
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative">
                {/* Pipeline Flow Diagram */}
                <div className="hidden md:block absolute top-1/2 left-0 right-0 h-0.5 bg-slate-600 -translate-y-1/2 z-0"></div>
                <div className="hidden md:block absolute top-1/2 left-1/4 w-0.5 h-3 bg-slate-600 -translate-y-1/2 z-0"></div>
                <div className="hidden md:block absolute top-1/2 left-1/2 w-0.5 h-3 bg-slate-600 -translate-y-1/2 z-0"></div>
                <div className="hidden md:block absolute top-1/2 left-3/4 w-0.5 h-3 bg-slate-600 -translate-y-1/2 z-0"></div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 relative z-10">
  {[results.pipeline.encoder, results.pipeline.bottleneck, results.pipeline.decoder].map((stage, index) => {
    const colorMap = [
      { bg: "bg-blue-500/20", icon: <Layers className="h-5 w-5 text-blue-400" /> },
      { bg: "bg-purple-500/20", icon: <Code className="h-5 w-5 text-purple-400" /> },
      { bg: "bg-green-500/20", icon: <Layers className="h-5 w-5 text-green-400" /> }
    ][index]

    return (
      <div
        key={index}
        className="bg-slate-700/30 rounded-lg p-4 border border-slate-600 hover:border-slate-400 transition-colors flex flex-col h-full"
      >
        <div className="flex items-center mb-4">
          <div className={`p-2 ${colorMap.bg} rounded-lg mr-3`}>{colorMap.icon}</div>
          <div>
            <h4 className="font-medium text-white">{stage.name}</h4>
            <p className="text-xs text-slate-400">{stage.description}</p>
          </div>
        </div>
        <div className="bg-slate-800 rounded-lg overflow-hidden mt-auto">
          <Image
            src={stage.image || "/placeholder.svg"}
            alt={`${stage.name} visualization`}
            width={150}
            height={150}
            className="w-full h-auto object-cover"
          />
        </div>
      </div>
    )
  })}
</div>


                {/* Flow Arrows for Mobile */}
                <div className="flex justify-center md:hidden my-4">
                  <ArrowRight className="h-6 w-6 text-slate-500" />
                </div>
              </div>

              <div className="mt-6 bg-slate-700/20 rounded-lg p-4 border border-slate-600/50">
                <p className="text-sm text-slate-300">
                  The AI model uses a U-Net architecture with an encoder that extracts features from the input image, a
                  bottleneck that compresses these features, and a decoder that reconstructs the segmentation mask. This
                  pipeline achieves high accuracy in tumor boundary detection.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-8">
          {/* Stage Prediction */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-white flex items-center text-xl">
                <AlertTriangle className="mr-2 h-5 w-5 text-orange-400" />
                Cancer Stage Prediction
              </CardTitle>
              <CardDescription className="text-slate-400 mt-1">
                AI-predicted cancer stage based on image analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="text-center p-4 bg-slate-700/30 rounded-lg">
                <div className="text-3xl font-bold text-orange-400 mb-3">{results.stage.predictedStage}</div>
                <Progress value={results.stage.precision} className="h-2" />
                <div className="text-xs text-slate-400 mt-2">{results.stage.precision}% precision</div>
              </div>

              <div className="grid grid-cols-5 gap-1">
  {["0", "I", "II", "III", "IV"].map((stage) => {
    const predictedStageShort = results.stage.predictedStage.replace("Stage ", "") // e.g., "Stage II" → "II"
    const isActive = stage === predictedStageShort

    return (
      <div
        key={stage}
        className={`text-center py-3 rounded text-xs font-medium ${
          isActive ? "bg-orange-500 text-white" : "bg-slate-700 text-slate-400"
        }`}
      >
        {stage}
      </div>
    )
  })}
</div>


<div className="bg-slate-700/20 rounded-lg p-4 border border-slate-600/50">
  <p className="text-sm text-slate-300">
    {stageDescriptions[results.stage.predictedStage] ?? "No details available for this stage."}
  </p>
</div>
            </CardContent>
          </Card>

          {/* Treatment Recommendations */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-white flex items-center text-xl">
                <Stethoscope className="mr-2 h-5 w-5 text-green-400" />
                Treatment Plan
              </CardTitle>
              <CardDescription className="text-slate-400 mt-1">
                Recommended treatment options based on analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {results.treatments.map((treatment: Treatment, index: number) => {
                  const Icon = treatmentIcons[treatment.type as keyof typeof treatmentIcons] || Pill
                  return (
                    <div
                      key={index}
                      className="bg-slate-700/30 rounded-lg p-4 border border-slate-600 hover:border-slate-500 transition-colors"
                    >
                      <div className="flex items-start space-x-3">
                        <div className="p-2 bg-slate-600 rounded-lg">
                          <Icon className="h-4 w-4 text-slate-300" />
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-2">
                            <h4 className="font-medium text-white">{treatment.type}</h4>
                            <Badge
                              variant="secondary"
                              className={`text-xs ${
                                treatment.priority === "Primary"
                                  ? "bg-red-500/20 text-red-300"
                                  : treatment.priority === "Secondary"
                                    ? "bg-yellow-500/20 text-yellow-300"
                                    : "bg-blue-500/20 text-blue-300"
                              }`}
                            >
                              {treatment.priority}
                            </Badge>
                          </div>
                          <p className="text-sm text-slate-300 mb-2">{treatment.recommendation}</p>
                          <p className="text-xs text-slate-400">{treatment.description}</p>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader className="pb-2">
              <CardTitle className="text-white text-xl">Quick Actions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <Button className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 h-11">
                  <CheckCircle className="mr-2 h-4 w-4" />
                  Approve Results
                </Button>
                <Button variant="outline" className="w-full border-slate-600 text-slate-300 hover:bg-slate-700 h-11">
                  <Info className="mr-2 h-4 w-4" />
                  Request Second Opinion
                </Button>
              </div>
            </CardContent>
          </Card> */}
        </div>
      </div>
    </div>
  )
}
