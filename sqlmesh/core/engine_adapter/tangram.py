from __future__ import annotations

import typing as t
from sqlglot import exp
from sqlmesh.core.engine_adapter.base import EngineAdapter
from sqlmesh.core.engine_adapter.clickhouse import ClickhouseEngineAdapter
from sqlmesh.core.engine_adapter.databricks import DatabricksEngineAdapter
from sqlmesh.core.engine_adapter.duckdb import DuckDBEngineAdapter
from sqlmesh.core.engine_adapter.postgres import PostgresEngineAdapter
from sqlmesh.core.engine_adapter.mixins import GetCurrentCatalogFromFunctionMixin
import pandas as pd
from sqlmesh.core.engine_adapter.shared import (
    CatalogSupport,
    DataObject,
    DataObjectType,
    SourceQuery,
    set_catalog,
)
from functools import partial


if t.TYPE_CHECKING:
    from sqlmesh.core._typing import SchemaName, TableName
    from sqlmesh.core.engine_adapter._typing import DF


class TangramSQLMixin(EngineAdapter):

    def fetchone(
        self,
        query: t.Union[exp.Expression, str],
        ignore_unsupported_errors: bool = False,
        quote_identifiers: bool = False,
    ) -> t.Tuple:
        self.execute(
            query,
            ignore_unsupported_errors=ignore_unsupported_errors,
            quote_identifiers=quote_identifiers,
        )
        return self.cursor.fetchone()

    def set_current_catalog(self, catalog: str) -> None:
        """Sets the catalog name of the current connection."""
        self.execute(f"use catalog {catalog}")

    def _df_to_source_queries(
        self,
        df: DF,
        columns_to_types: t.Dict[str, exp.DataType],
        batch_size: int,
        target_table: TableName,
    ) -> t.List[SourceQuery]:
        assert isinstance(df, pd.DataFrame)
        num_rows = len(df.index)
        batch_size = sys.maxsize if batch_size == 0 else batch_size
        values = list(df.itertuples(index=False, name=None))
        return [
            SourceQuery(
                query_factory=partial(
                    self._values_to_sql,
                    values=values,
                    columns_to_types=columns_to_types,
                    batch_start=i,
                    batch_end=min(i + batch_size, num_rows),
                ),
            )
            for i in range(0, num_rows, batch_size)
        ]

    def _get_data_objects(
        self, schema_name: SchemaName, object_names: t.Optional[t.Set[str]] = None
    ) -> t.List[DataObject]:
        """
        Returns all the data objects that exist in the given schema and optionally catalog.
        """
        catalog = self.get_current_catalog()

        if isinstance(schema_name, exp.Table):
            # Ensures we don't generate identifier quotes
            schema_name = ".".join(part.name for part in schema_name.parts)

        query = f"show tables in {catalog}.{schema_name}"
        rows = self.fetchall(query)
        # Get column names from cursor description
        df = pd.DataFrame(
            rows, columns=['table_name', 'table_type', 'app', 'rn', 'registered_at'])
        return [
            DataObject(
                catalog=catalog,  # type: ignore
                schema=schema_name,  # type: ignore
                name=row.table_name,  # type: ignore
                type=DataObjectType.from_str(row.table_type),  # type: ignore
            )
            for row in df.itertuples()
        ]


@set_catalog(override_mapping={"_get_data_objects": CatalogSupport.REQUIRES_SET_CATALOG})
class TangramDuckDBEngineAdapter(TangramSQLMixin, DuckDBEngineAdapter):
    pass


@set_catalog(
    {
        "_get_data_objects": CatalogSupport.REQUIRES_SET_CATALOG,
    }
)
class TangramDatabricksEngineAdapter(TangramSQLMixin, DatabricksEngineAdapter):
    pass


@set_catalog(
    {
        "_get_data_objects": CatalogSupport.REQUIRES_SET_CATALOG,
    }
)
class TangramClickhouseEngineAdapter(TangramSQLMixin, GetCurrentCatalogFromFunctionMixin, ClickhouseEngineAdapter):
    @property
    def catalog_support(self) -> CatalogSupport:
        return CatalogSupport.FULL_SUPPORT


@set_catalog(
    {
        "_get_data_objects": CatalogSupport.REQUIRES_SET_CATALOG,
    }
)
class TangramPostgresEngineAdapter(TangramSQLMixin, PostgresEngineAdapter):
    pass
