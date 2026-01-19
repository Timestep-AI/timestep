type MlflowConfig = {
  trackingUri: string;
  experimentName: string;
};

type MlflowTag = { key: string; value: string };
type MlflowParam = { key: string; value: string };
type MlflowMetric = { key: string; value: number; timestamp: number; step: number };

type LogBatchPayload = {
  run_id: string;
  metrics?: MlflowMetric[];
  params?: MlflowParam[];
  tags?: MlflowTag[];
};

type CreateRunResponse = {
  run: {
    info: {
      run_id: string;
    };
  };
};

type GetExperimentResponse = {
  experiment: {
    experiment_id: string;
  };
};

type CreateExperimentResponse = {
  experiment_id: string;
};

function getMlflowConfig(): MlflowConfig | null {
  const trackingUri = process.env.MLFLOW_TRACKING_URI;
  if (!trackingUri || !(trackingUri.startsWith('http://') || trackingUri.startsWith('https://'))) {
    return null;
  }
  return {
    trackingUri,
    experimentName: process.env.MLFLOW_EXPERIMENT_NAME || 'timestep-evals',
  };
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`MLflow request failed (${response.status}): ${text}`);
  }
  return (await response.json()) as T;
}

async function getOrCreateExperimentId(trackingUri: string, experimentName: string): Promise<string> {
  try {
    const data = await postJson<GetExperimentResponse>(`${trackingUri}/api/2.0/mlflow/experiments/get-by-name`, {
      experiment_name: experimentName,
    });
    return data.experiment.experiment_id;
  } catch (error) {
    const created = await postJson<CreateExperimentResponse>(`${trackingUri}/api/2.0/mlflow/experiments/create`, {
      name: experimentName,
    });
    return created.experiment_id;
  }
}

export class MlflowClient {
  private trackingUri: string;
  private experimentId: string;

  private constructor(trackingUri: string, experimentId: string) {
    this.trackingUri = trackingUri;
    this.experimentId = experimentId;
  }

  static async create(): Promise<MlflowClient | null> {
    const config = getMlflowConfig();
    if (!config) {
      return null;
    }
    const experimentId = await getOrCreateExperimentId(config.trackingUri, config.experimentName);
    return new MlflowClient(config.trackingUri, experimentId);
  }

  async createRun(runName: string, tags: Record<string, string>): Promise<string> {
    const tagList: MlflowTag[] = [
      { key: 'mlflow.runName', value: runName },
      ...Object.entries(tags).map(([key, value]) => ({ key, value })),
    ];
    const data = await postJson<CreateRunResponse>(`${this.trackingUri}/api/2.0/mlflow/runs/create`, {
      experiment_id: this.experimentId,
      tags: tagList,
    });
    return data.run.info.run_id;
  }

  async logBatch(runId: string, payload: { params?: Record<string, string>; metrics?: Record<string, number>; tags?: Record<string, string> }): Promise<void> {
    const timestamp = Date.now();
    const params = payload.params
      ? Object.entries(payload.params).map(([key, value]) => ({ key, value: String(value) }))
      : undefined;
    const metrics = payload.metrics
      ? Object.entries(payload.metrics).map(([key, value]) => ({ key, value: Number(value), timestamp, step: 0 }))
      : undefined;
    const tags = payload.tags
      ? Object.entries(payload.tags).map(([key, value]) => ({ key, value }))
      : undefined;

    const body: LogBatchPayload = {
      run_id: runId,
      ...(params ? { params } : {}),
      ...(metrics ? { metrics } : {}),
      ...(tags ? { tags } : {}),
    };

    if (!params && !metrics && !tags) {
      return;
    }
    await postJson(`${this.trackingUri}/api/2.0/mlflow/runs/log-batch`, body);
  }
}

let cachedClient: Promise<MlflowClient | null> | null = null;

export async function getMlflowClient(): Promise<MlflowClient | null> {
  if (!cachedClient) {
    cachedClient = MlflowClient.create().catch((error) => {
      console.error('MLflow client init failed:', error);
      return null;
    });
  }
  return cachedClient;
}
