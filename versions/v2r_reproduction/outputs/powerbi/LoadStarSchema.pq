let
    FolderPath = "REPLACE_WITH_ABSOLUTE_PATH_TO_outputs/powerbi",
    Fact = Table.PromoteHeaders(Csv.Document(File.Contents(FolderPath & "/fact_indicator_value.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv])),
    DimIndicator = Table.PromoteHeaders(Csv.Document(File.Contents(FolderPath & "/dim_indicator.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv])),
    DimCountry = Table.PromoteHeaders(Csv.Document(File.Contents(FolderPath & "/dim_country.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv])),
    DimPhase = Table.PromoteHeaders(Csv.Document(File.Contents(FolderPath & "/dim_phase.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv])),
    DimSource = Table.PromoteHeaders(Csv.Document(File.Contents(FolderPath & "/dim_source.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]))
in
    Fact
