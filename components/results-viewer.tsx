"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
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
} from "lucide-react"
import Image from "next/image"

interface ResultsViewerProps {
  results: any
  originalImage: string
  fileName: string
  onNewAnalysis: () => void
}

export function ResultsViewer({ results, originalImage, fileName, onNewAnalysis }: ResultsViewerProps) {
  const [sliderValue, setSliderValue] = useState([50])
  const [showExplainability, setShowExplainability] = useState(false)
  const [brightness, setBrightness] = useState(100)
  const [contrast, setContrast] = useState(100)

  const treatmentIcons = {
    Surgery: Scissors,
    Chemotherapy: Pill,
    Radiation: Radiation,
    "Hormone Therapy": Heart,
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Analysis Results</h1>
          <p className="text-slate-400">
            File: {fileName} | Analyzed: {new Date().toLocaleDateString()}
          </p>
        </div>
        <div className="flex space-x-3">
          <Button variant="outline" className="border-slate-600 text-slate-300 hover:bg-slate-800">
            <Download className="mr-2 h-4 w-4" />
            Export Report
          </Button>
          <Button variant="outline" className="border-slate-600 text-slate-300 hover:bg-slate-800">
            <Share2 className="mr-2 h-4 w-4" />
            Share
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
        <div className="lg:col-span-2 space-y-6">
          {/* Image Comparison */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-white flex items-center">
                  <Eye className="mr-2 h-5 w-5" />
                  Segmentation Results
                </CardTitle>
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
            <CardContent>
              <div className="relative bg-slate-900 rounded-lg overflow-hidden mb-4">
                <div className="flex">
                  <div className="w-1/2 relative">
                    <Image
                      src={originalImage || "/placeholder.svg"}
                      alt="Original Image"
                      width={400}
                      height={400}
                      className="w-full h-64 object-cover"
                      style={{
                        filter: `brightness(${brightness}%) contrast(${contrast}%)`,
                      }}
                    />
                    <div className="absolute top-2 left-2">
                      <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30">Original</Badge>
                    </div>
                  </div>
                  <div className="w-1/2 relative">
                    <Image
                      src="/placeholder.svg?height=400&width=400"
                      alt="Segmented Image"
                      width={400}
                      height={400}
                      className="w-full h-64 object-cover"
                      style={{
                        filter: `brightness(${brightness}%) contrast(${contrast}%)`,
                      }}
                    />
                    <div className="absolute top-2 left-2">
                      <Badge className="bg-purple-500/20 text-purple-300 border-purple-500/30">Segmented</Badge>
                    </div>
                    {showExplainability && (
                      <div className="absolute inset-0 bg-gradient-to-r from-red-500/30 to-yellow-500/30 mix-blend-overlay" />
                    )}
                  </div>
                </div>

                {/* Slider Overlay */}
                <div className="absolute inset-x-0 bottom-4 px-4">
                  <Slider value={sliderValue} onValueChange={setSliderValue} max={100} step={1} className="w-full" />
                </div>
              </div>

              {/* Brightness and Contrast Controls */}
              <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600 mb-4">
                <h4 className="text-white font-medium mb-4 flex items-center">
                  <Eye className="mr-2 h-4 w-4 text-blue-400" />
                  Image Controls
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
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
                  <div className="space-y-2">
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
                <div className="flex justify-between mt-4">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setBrightness(100)
                      setContrast(100)
                    }}
                    className="border-slate-600 text-slate-300 hover:bg-slate-700"
                  >
                    Reset
                  </Button>
                  <div className="flex space-x-2">
                    <Button
                      variant="outline"
                      size="sm"
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
                      size="sm"
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
              </div>

              {showExplainability && (
                <div className="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
                  <h4 className="text-white font-medium mb-2 flex items-center">
                    <Zap className="mr-2 h-4 w-4 text-yellow-400" />
                    AI Explainability (Grad-CAM)
                  </h4>
                  <p className="text-slate-300 text-sm">
                    The heatmap overlay shows regions that most influenced the AI's decision. Red areas indicate high
                    attention, while yellow areas show moderate attention.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Detailed Metrics */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Activity className="mr-2 h-5 w-5" />
                Segmentation Metrics
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-400 mb-1">{results.segmentation.diceScore}%</div>
                  <div className="text-sm text-slate-400">Dice Score</div>
                  <Progress value={results.segmentation.diceScore} className="mt-2 h-2" />
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-400 mb-1">{results.segmentation.confidence}%</div>
                  <div className="text-sm text-slate-400">Confidence</div>
                  <Progress value={results.segmentation.confidence} className="mt-2 h-2" />
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-400 mb-1">{results.segmentation.tumorSize}</div>
                  <div className="text-sm text-slate-400">Tumor Size</div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
                <div className="bg-slate-700/50 rounded-lg p-3">
                  <div className="text-sm text-slate-400">BI-RADS Score</div>
                  <div className="text-lg font-semibold text-orange-400">{results.segmentation.biRadsScore}</div>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-3">
                  <div className="text-sm text-slate-400">Margin</div>
                  <div className="text-lg font-semibold text-red-400">{results.segmentation.margin}</div>
                </div>
                <div className="bg-slate-700/50 rounded-lg p-3">
                  <div className="text-sm text-slate-400">Density</div>
                  <div className="text-lg font-semibold text-yellow-400">{results.segmentation.density}</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Stage Prediction */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <AlertTriangle className="mr-2 h-5 w-5 text-orange-400" />
                Cancer Stage Prediction
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-center mb-4">
                <div className="text-3xl font-bold text-orange-400 mb-2">{results.stage.predictedStage}</div>
                <div className="text-sm text-slate-400 mb-2">TNM: {results.stage.tnmStaging}</div>
                <Progress value={results.stage.confidence} className="h-2" />
                <div className="text-xs text-slate-400 mt-1">{results.stage.confidence}% confidence</div>
              </div>

              <div className="grid grid-cols-5 gap-1 mb-4">
                {["0", "I", "II", "III", "IV"].map((stage) => (
                  <div
                    key={stage}
                    className={`text-center py-2 rounded text-xs font-medium ${
                      stage === "II" ? "bg-orange-500 text-white" : "bg-slate-700 text-slate-400"
                    }`}
                  >
                    {stage}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Treatment Recommendations */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Stethoscope className="mr-2 h-5 w-5 text-green-400" />
                Treatment Plan
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {results.treatments.map((treatment: any, index: number) => {
                  const Icon = treatmentIcons[treatment.type as keyof typeof treatmentIcons] || Pill
                  return (
                    <div
                      key={index}
                      className="bg-slate-700/50 rounded-lg p-4 border border-slate-600 hover:border-slate-500 transition-colors"
                    >
                      <div className="flex items-start space-x-3">
                        <div className="p-2 bg-slate-600 rounded-lg">
                          <Icon className="h-4 w-4 text-slate-300" />
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center justify-between mb-1">
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

          {/* Quick Actions */}
          <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
            <CardHeader>
              <CardTitle className="text-white">Quick Actions</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <Button className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700">
                  <CheckCircle className="mr-2 h-4 w-4" />
                  Approve Results
                </Button>
                <Button variant="outline" className="w-full border-slate-600 text-slate-300 hover:bg-slate-700">
                  <Info className="mr-2 h-4 w-4" />
                  Request Second Opinion
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
