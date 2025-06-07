"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import {
  Settings,
  Shield,
  Moon,
  Sun,
  Bell,
  Lock,
  Database,
  FileText,
  Mail,
  Phone,
  MapPin,
  CheckCircle,
  AlertTriangle,
  Info,
} from "lucide-react"
import { Navigation } from "@/components/navigation"
import { useRouter } from "next/navigation"

export default function SettingsPage() {
  const [darkMode, setDarkMode] = useState(true)
  const [notifications, setNotifications] = useState(true)
  const [autoSave, setAutoSave] = useState(true)
  const [dataRetention, setDataRetention] = useState(true)

  const complianceItems = [
    {
      title: "HIPAA Compliance",
      status: "Active",
      description: "Health Insurance Portability and Accountability Act",
      icon: Shield,
      color: "green",
    },
    {
      title: "GDPR Compliance",
      status: "Active",
      description: "General Data Protection Regulation",
      icon: Lock,
      color: "green",
    },
    {
      title: "FDA 510(k)",
      status: "Pending",
      description: "Medical Device Clearance",
      icon: FileText,
      color: "yellow",
    },
    {
      title: "ISO 27001",
      status: "Active",
      description: "Information Security Management",
      icon: Database,
      color: "green",
    },
  ]
  const router = useRouter()
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <Navigation
  onReset={() => {
    router.push("/")
  }}
  currentState="results"
/>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Settings</h1>
          <p className="text-slate-400">Configure system preferences and compliance settings</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Settings */}
          <div className="lg:col-span-2 space-y-6">
            {/* General Settings */}
            <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-white flex items-center">
                  <Settings className="mr-2 h-5 w-5" />
                  General Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      {darkMode ? (
                        <Moon className="h-4 w-4 text-slate-400" />
                      ) : (
                        <Sun className="h-4 w-4 text-slate-400" />
                      )}
                      <span className="text-white font-medium">Dark Mode</span>
                    </div>
                    <p className="text-sm text-slate-400">Use dark theme for reduced eye strain during long sessions</p>
                  </div>
                  <Switch checked={darkMode} onCheckedChange={setDarkMode} />
                </div>

                <Separator className="bg-slate-700" />

                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <Bell className="h-4 w-4 text-slate-400" />
                      <span className="text-white font-medium">Notifications</span>
                    </div>
                    <p className="text-sm text-slate-400">Receive alerts for analysis completion and system updates</p>
                  </div>
                  <Switch checked={notifications} onCheckedChange={setNotifications} />
                </div>

                <Separator className="bg-slate-700" />

                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <Database className="h-4 w-4 text-slate-400" />
                      <span className="text-white font-medium">Auto-Save Results</span>
                    </div>
                    <p className="text-sm text-slate-400">Automatically save analysis results to secure storage</p>
                  </div>
                  <Switch checked={autoSave} onCheckedChange={setAutoSave} />
                </div>

                <Separator className="bg-slate-700" />

                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center space-x-2">
                      <Lock className="h-4 w-4 text-slate-400" />
                      <span className="text-white font-medium">Data Retention</span>
                    </div>
                    <p className="text-sm text-slate-400">Retain patient data according to institutional policies</p>
                  </div>
                  <Switch checked={dataRetention} onCheckedChange={setDataRetention} />
                </div>
              </CardContent>
            </Card>

            {/* Compliance Status */}
            <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-white flex items-center">
                  <Shield className="mr-2 h-5 w-5 text-green-400" />
                  Compliance & Security
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {complianceItems.map((item, index) => {
                    const Icon = item.icon
                    return (
                      <div key={index} className="bg-slate-700/50 rounded-lg p-4 border border-slate-600">
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex items-center space-x-2">
                            <Icon className="h-5 w-5 text-slate-400" />
                            <span className="font-medium text-white">{item.title}</span>
                          </div>
                          <Badge
                            variant="secondary"
                            className={`${
                              item.color === "green"
                                ? "bg-green-500/20 text-green-300"
                                : "bg-yellow-500/20 text-yellow-300"
                            }`}
                          >
                            {item.status === "Active" ? (
                              <CheckCircle className="mr-1 h-3 w-3" />
                            ) : (
                              <AlertTriangle className="mr-1 h-3 w-3" />
                            )}
                            {item.status}
                          </Badge>
                        </div>
                        <p className="text-sm text-slate-400">{item.description}</p>
                      </div>
                    )
                  })}
                </div>

                <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                  <div className="flex items-start space-x-3">
                    <Info className="h-5 w-5 text-blue-400 mt-0.5" />
                    <div>
                      <h4 className="text-blue-300 font-medium mb-1">Security Notice</h4>
                      <p className="text-sm text-blue-200">
                        All patient data is encrypted at rest and in transit. Access logs are maintained for audit
                        purposes. Regular security assessments are conducted.
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* System Info */}
            <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-white">System Information</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <div className="text-sm text-slate-400">Version</div>
                  <div className="text-white font-medium">OncoPath v2.1.0</div>
                </div>
                <div>
                  <div className="text-sm text-slate-400">Last Updated</div>
                  <div className="text-white font-medium">June 3, 2025</div>
                </div>
                <div>
                  <div className="text-sm text-slate-400">License</div>
                  <div className="text-white font-medium">Enterprise</div>
                </div>
                <div>
                  <div className="text-sm text-slate-400">Support Level</div>
                  <div className="text-white font-medium">24/7 Premium</div>
                </div>
              </CardContent>
            </Card>

            {/* Contact Information */}
            <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-white">Support Contact</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-3">
                  <Mail className="h-4 w-4 text-slate-400" />
                  <div>
                    <div className="text-sm text-slate-400">Email</div>
                    <div className="text-white text-sm">support@oncopath.ai</div>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <Phone className="h-4 w-4 text-slate-400" />
                  <div>
                    <div className="text-sm text-slate-400">Phone</div>
                    <div className="text-white text-sm">+91 9108254345</div>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  <MapPin className="h-4 w-4 text-slate-400" />
                  <div>
                    <div className="text-sm text-slate-400">Address</div>
                    <div className="text-white text-sm">
                      123 Medical AI Road
                      <br />
                      Bengaluru, IN
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <Card className="bg-slate-800/50 border-slate-700 backdrop-blur-sm">
              <CardHeader>
                <CardTitle className="text-white">Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Button variant="outline" className="w-full border-slate-600 text-slate-300 hover:bg-slate-700">
                  <FileText className="mr-2 h-4 w-4" />
                  Download Compliance Report
                </Button>
                <Button variant="outline" className="w-full border-slate-600 text-slate-300 hover:bg-slate-700">
                  <Database className="mr-2 h-4 w-4" />
                  Export System Logs
                </Button>
                <Button variant="outline" className="w-full border-slate-600 text-slate-300 hover:bg-slate-700">
                  <Shield className="mr-2 h-4 w-4" />
                  Security Audit
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
