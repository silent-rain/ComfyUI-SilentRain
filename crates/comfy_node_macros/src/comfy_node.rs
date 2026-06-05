//! This crate provides the `#[comfy_node]` attribute macro that automatically
//! generates the necessary `#[pyclass]` and `#[pymethods]` boilerplate
//! for ComfyUI v3 nodes written in pure Rust.
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse_macro_input;

/// Attribute macro for ComfyUI v3 node structs.
pub fn comfy_node_impl(_args: TokenStream, input: TokenStream) -> TokenStream {
    // Parse the struct definition
    let struct_def: syn::ItemStruct = parse_macro_input!(input as syn::ItemStruct);

    // Get the original struct name
    let struct_name = &struct_def.ident;

    // Generate the expanded code
    // Note: In pyo3 0.28, #[pymethods] classmethods receive:
    // - cls: &Bound<'_, PyType>  (for execute/check_lazy_status/fingerprint_inputs)
    // - cls: Bound<'py, PyType>  (for define_schema)
    // - py: Python<'py>
    // - args: &Bound<'py, PyTuple>
    // - kwargs: Option<Bound<'_, PyDict>>
    let expanded: TokenStream2 = quote! {
        // ------------------------------------------------------------------
        // 1. Original struct with #[pyclass] added
        // ------------------------------------------------------------------
        #struct_def

        // ------------------------------------------------------------------
        // 2. Generate the #[pymethods] impl block for Python interop
        // ------------------------------------------------------------------
        #[pymethods]
        impl #struct_name {
            /// Python constructor — calls ComfyNode::new()
            #[new]
            fn new() -> Self {
                <#struct_name as ComfyNode>::new()
            }

            #[classmethod]
            fn define_schema<'py>(
                _cls: Bound<'py, PyType>,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyAny>> {
                // define_schema now returns Result<NodeSchema, Error>
                // Use ? to automatically convert Error to PyErr (via From<Error> for PyErr)
                let schema = <#struct_name as ComfyNode>::define_schema()?;
                schema.into_py_schema(py)
            }

            #[classmethod]
            #[pyo3(name = "execute", signature = (*args, **kwargs))]
            fn execute<'py>(
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                args: &Bound<'py, PyTuple>,
                kwargs: Option<Bound<'py, PyDict>>,
            ) -> PyResult<Bound<'py, PyAny>> {
                // execute now returns Result<NodeOutput, Error>
                // Use ? to automatically convert Error to PyErr
                let result = <#struct_name as ComfyNode>::execute(py, args, kwargs)?;
                result.to_py_obj(py)
            }

            #[classmethod]
            #[pyo3(name = "validate_inputs", signature = (*args, **kwargs))]
            fn validate_inputs<'py>(
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                args: &Bound<'py, PyTuple>,
                kwargs: Option<Bound<'py, PyDict>>,
            ) -> PyResult<Bound<'py, PyAny>> {
                // validate_inputs is now a static method
                // Use ? to automatically convert Error to PyErr
                <#struct_name as ComfyNode>::validate_inputs(py, args, kwargs)?;
                // Return True for successful validation
                Ok(PyBool::new(py, true).as_any().clone())
            }

            #[classmethod]
            #[pyo3(name = "fingerprint_inputs", signature = (*args, **kwargs))]
            fn fingerprint_inputs<'py>(
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                args: &Bound<'py, PyTuple>,
                kwargs: Option<Bound<'py, PyDict>>,
            ) -> PyResult<Option<String>> {
                // fingerprint_inputs is now a static method
                // Use ? to automatically convert Error to PyErr
                let result = <#struct_name as ComfyNode>::fingerprint_inputs(py, args, kwargs)?;
                Ok(result)
            }

            #[classmethod]
            #[pyo3(name = "check_lazy_status", signature = (*args, **kwargs))]
            fn check_lazy_status<'py>(
                _cls: &Bound<'_, PyType>,
                py: Python<'py>,
                args: &Bound<'py, PyTuple>,
                kwargs: Option<Bound<'py, PyDict>>,
            ) -> PyResult<Vec<String>> {
                // check_lazy_status is now a static method
                // Use ? to automatically convert Error to PyErr
                let result = <#struct_name as ComfyNode>::check_lazy_status(py, args, kwargs)?;
                Ok(result)
            }

        }
    };

    TokenStream::from(expanded)
}
