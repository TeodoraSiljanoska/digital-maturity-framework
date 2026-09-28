# correlation_vif_process

```mermaid
flowchart LR
  Panel[Lagged analysis panel] --> Corr[Pearson correlation]
  Corr --> VIF[VIF with intercept]
  VIF --> Fig[Heatmap with caption]
```
