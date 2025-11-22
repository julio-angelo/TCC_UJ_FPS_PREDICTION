import pandas as pd

# --- 1. Defina os nomes dos arquivos ---
# !!! ALTERE ESTES NOMES para corresponderem aos seus arquivos !!!
input_csv_path = '360_test.csv'
output_csv_path = '360_test.csv'


# --- 2. Bloco principal para ler, corrigir, filtrar, remover linhas e salvar ---
try:
    # --- ETAPA DE CORREÇÃO (para colunas sem nome) ---
    print("Iniciando correção: Lendo apenas o cabeçalho...")
    header_df = pd.read_csv(input_csv_path, nrows=0, on_bad_lines='warn')
    all_column_names = header_df.columns.tolist()

    valid_columns = [col for col in all_column_names if not str(col).startswith('Unnamed:')]
    removed_columns_count = len(all_column_names) - len(valid_columns)

    if removed_columns_count > 0:
        print(f"Detectadas e ignoradas {removed_columns_count} colunas sem nome.")

    print(f"Lendo o arquivo completo, carregando apenas as {len(valid_columns)} colunas válidas...")
    df = pd.read_csv(input_csv_path, usecols=valid_columns, on_bad_lines='warn')
    print("Arquivo lido e corrigido com sucesso!")
    print(f"Número de linhas carregadas inicialmente: {len(df)}")

    # --- EMOÇÃO DAS ÚLTIMAS 10 LINHAS ---
    if len(df) > 10:
        print("\nRemovendo as últimas 10 linhas dos dados carregados...")
        df = df.iloc[:-10]
        print(f"Número de linhas após a remoção: {len(df)}")
    else:
        print("\nO arquivo tem 10 linhas ou menos, nenhuma linha foi removida.")


    # --- ETAPA DE FILTRAGEM (para manter apenas as colunas desejadas) ---
    columns_to_keep_str = """
    Exclusive/AllWorkers/RenderPrePass	Exclusive/AllWorkers/RenderPostProcessing	Exclusive/AllWorkers/UpdateGPUScene	Exclusive/AllWorkers/RenderTranslucency	Exclusive/AllWorkers/Slate	Exclusive/AllWorkers/RenderBasePass	Exclusive/AllWorkers/SortLights	Exclusive/AllWorkers/ComputeLightGrid	Exclusive/AllWorkers/RenderShadows	Exclusive/AllWorkers/RenderLighting	TransientMemoryAliasedMB	GPUMem/LocalBudgetMB	GPUMem/LocalUsedMB	GPUMem/SystemBudgetMB	GPUMem/SystemUsedMB	RHI/DrawCalls	RHI/PrimitivesDrawn	Exclusive/AllWorkers/Physics	Exclusive/AllWorkers/Material_UpdateDeferredCachedUniformExpressions	Scheduler/AllWorkers/SignalStandbyThread	Exclusive/RenderThread/RenderThreadOther	Exclusive/RenderThread/RemovePrimitiveSceneInfos	Exclusive/RenderThread/UpdatePrimitiveInstances	Exclusive/RenderThread/ConsolidateInstanceDataAllocations	Exclusive/RenderThread/AddPrimitiveSceneInfos	Exclusive/RenderThread/UpdatePrimitiveTransform	Exclusive/RenderThread/PreRender	Exclusive/RenderThread/UpdateGPUScene	Exclusive/RenderThread/PrepareDistanceFieldScene	Exclusive/RenderThread/RenderOther	Exclusive/RenderThread/InitViews_Scene	Exclusive/RenderThread/Niagara	Exclusive/RenderThread/FXSystem	Exclusive/RenderThread/GPUSort	Exclusive/RenderThread/RenderPrePass	Exclusive/RenderThread/RenderVelocities	Exclusive/RenderThread/SortLights	Exclusive/RenderThread/ComputeLightGrid	Exclusive/RenderThread/DeferredShadingSceneRenderer_DBuffer	Exclusive/RenderThread/RenderBasePass	Exclusive/RenderThread/RenderShadows	Exclusive/RenderThread/AfterBasePass	Exclusive/RenderThread/RenderLighting	Exclusive/RenderThread/RenderFog	Exclusive/RenderThread/RenderLocalFogVolume	Exclusive/RenderThread/RenderOpaqueFX	Exclusive/RenderThread/RenderTranslucency	Exclusive/RenderThread/RenderPostProcessing	Exclusive/RenderThread/RDG	Exclusive/RenderThread/STAT_RDG_FlushResourcesRHI	Exclusive/RenderThread/RDG_CollectResources	Exclusive/RenderThread/RDG_Execute	Exclusive/RenderThread/PostRenderCleanUp	Exclusive/RenderThread/Material_UpdateDeferredCachedUniformExpressions	Exclusive/RenderThread/Slate	DrawSceneCommand_StartDelay	GPUSceneInstanceCount	LightCount/All	RayTracingGeometry/RequestedSizeMB	RayTracingGeometry/TotalResidentSizeMB	RayTracingGeometry/TotalAlwaysResidentSizeMB	RenderTargetPoolUsed	RenderTargetPoolCount	RDGCount/Passes	RDGCount/Buffers	RDGCount/Textures	RenderThreadIdle/Total	RenderThreadIdle/CriticalPath	RenderThreadIdle/SwapBuffer	RenderThreadIdle/NonCriticalPath	RenderTargetProfiler/Total	TextureProfiler/Total	TextureProfiler/Other	Exclusive/GameThread/Input	Exclusive/GameThread/TimerManager	Exclusive/GameThread/AsyncLoading	Exclusive/GameThread/WorldTickMisc	Exclusive/GameThread/Effects	Exclusive/GameThread/NetworkIncoming	NavigationBuildDetailed/GameThread/Navigation_RebuildDirtyAreas	NavigationBuildDetailed/GameThread/Navigation_TickAsyncBuild	NavigationBuildDetailed/GameThread/Navigation_CrowdManager	Exclusive/GameThread/NavigationBuild	Exclusive/GameThread/ResetAsyncTraceTickTime	Exclusive/GameThread/WorldPreActorTick	Exclusive/GameThread/QueueTicks	Exclusive/GameThread/TickActors	Exclusive/GameThread/Physics	Exclusive/GameThread/SyncBodies	Exclusive/GameThread/FlushLatentActions	Exclusive/GameThread/Tickables	Exclusive/GameThread/EnvQueryManager	Exclusive/GameThread/Landscape	Exclusive/GameThread/Camera	Exclusive/GameThread/RecordTickCountsToCSV	Exclusive/GameThread/RecordActorCountsToCSV	Exclusive/GameThread/Audio	Exclusive/GameThread/EndOfFrameUpdates	Exclusive/GameThread/DebugHUD	Exclusive/GameThread/RenderAssetStreaming	Slate/GameThread/TickPlatform	Exclusive/GameThread/UI	Slate/GameThread/DrawPrePass	Slate/GameThread/DrawWindows_Private	Exclusive/GameThread/DeferredTickTime	Exclusive/GameThread/CsvProfiler	Exclusive/GameThread/LLM	Ticks/Total	ActorCount/TotalActorCount	TextureStreaming/StreamingPool	TextureStreaming/CachedMips	TextureStreaming/WantedMips	TextureStreaming/NonStreamingMips	Shaders/ShaderMemoryMB	Shaders/NumShadersLoaded	Shaders/NumShaderMaps	Shaders/NumShadersCreated	Shaders/NumShaderMapsUsedForRendering	FrameTime	MemoryFreeMB	PhysicalUsedMB	VirtualUsedMB	SystemMaxMB	GPUTime	RHIThreadTime	InputLatencyTime	CPUUsage_Process	CPUUsage_Idle
    """
    columns_to_keep = [col.strip() for col in columns_to_keep_str.strip().split()]
    
    final_columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    
    print(f"\nDas suas colunas desejadas, {len(final_columns_to_keep)} foram encontradas no arquivo.")
    
    df_final = df[final_columns_to_keep]

    # Salva o resultado final em um novo arquivo CSV.
    print(f"Salvando os dados finais em: {output_csv_path}...")
    df_final.to_csv(output_csv_path, index=False)

    print("\nOperação concluída com sucesso!")
    print(f"O arquivo '{output_csv_path}' foi criado com {df_final.shape[0]} linhas e {df_final.shape[1]} colunas.")

except FileNotFoundError:
    print(f"ERRO: O arquivo '{input_csv_path}' não foi encontrado. Verifique o nome e o caminho do arquivo.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")
