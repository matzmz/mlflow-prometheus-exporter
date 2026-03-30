{{- define "mlflow-prometheus-exporter.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "mlflow-prometheus-exporter.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name (include "mlflow-prometheus-exporter.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "mlflow-prometheus-exporter.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" -}}
{{- end -}}

{{- define "mlflow-prometheus-exporter.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mlflow-prometheus-exporter.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "mlflow-prometheus-exporter.labels" -}}
helm.sh/chart: {{ include "mlflow-prometheus-exporter.chart" . }}
{{ include "mlflow-prometheus-exporter.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end -}}

{{- define "mlflow-prometheus-exporter.secretName" -}}
{{- if .Values.mlflow.auth.existingSecret -}}
{{- .Values.mlflow.auth.existingSecret -}}
{{- else -}}
{{- include "mlflow-prometheus-exporter.fullname" . -}}
{{- end -}}
{{- end -}}

{{- define "mlflow-prometheus-exporter.image" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end -}}
