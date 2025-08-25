#!/usr/bin/env python3
"""
REINVENT4 Streamlit Web Application - Part 2
Additional pages and functionality for the comprehensive REINVENT4 GUI.
"""

def show_rgroup_page():
    """R-Group replacement page using LibInvent"""
    
    st.markdown('<div class="sub-header">⚗️ R-Group Replacement</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Replace specific R-groups in molecules while keeping the core scaffold intact.
    </div>
    """, unsafe_allow_html=True)
    
    # Configuration
    with st.expander("⚙️ Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Configuration")
            
            model_file = st.text_input(
                "LibInvent Model File",
                value="priors/libinvent.prior",
                help="Path to the trained LibInvent model"
            )
            
            device = st.selectbox("Compute Device", ["cuda:0", "cpu"])
            
            num_variants = st.number_input(
                "Number of Variants per Molecule",
                min_value=1,
                max_value=200,
                value=30
            )
        
        with col2:
            st.subheader("Input Molecules")
            
            input_method = st.radio(
                "Input Method:",
                ["Upload File", "Text Input", "Draw Structure"]
            )
            
            molecules = []
            
            if input_method == "Upload File":
                uploaded_file = st.file_uploader(
                    "Upload Molecule File",
                    type=['smi', 'sdf', 'txt', 'csv'],
                    help="File containing molecules with R-groups marked"
                )
                
                if uploaded_file:
                    content = uploaded_file.read().decode('utf-8')
                    molecules = [line.strip() for line in content.split('\n') if line.strip()]
            
            elif input_method == "Text Input":
                molecules_text = st.text_area(
                    "Enter Molecules (one per line)",
                    placeholder="c1ccc([*])cc1\nCC([*])C([*])=O\n...",
                    height=150,
                    help="Enter molecules with [*] marking R-group positions"
                )
                
                if molecules_text:
                    molecules = [line.strip() for line in molecules_text.split('\n') if line.strip()]
            
            else:  # Draw Structure
                st.info("Structure drawing functionality would be integrated here using rdkit-js or similar")
                
                # Placeholder for structure drawing
                example_mol = st.selectbox(
                    "Select Example Molecule:",
                    [
                        "c1ccc([*])cc1",  # phenyl with R-group
                        "CC([*])C=O",     # acetyl with R-group
                        "c1ccnc([*])c1",  # pyridyl with R-group
                    ]
                )
                
                if example_mol:
                    molecules = [example_mol]
    
    # R-group constraints
    with st.expander("🧪 R-Group Constraints"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Chemical Constraints")
            
            allowed_elements = st.multiselect(
                "Allowed Elements in R-groups",
                ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"],
                default=["C", "N", "O", "S", "F", "Cl", "Br"],
                help="Limit elements that can appear in R-groups"
            )
            
            max_rgroup_size = st.number_input(
                "Maximum R-group Size (heavy atoms)",
                min_value=1,
                max_value=20,
                value=8
            )
            
            allow_rings_in_rgroup = st.checkbox(
                "Allow Rings in R-groups",
                value=True
            )
        
        with col2:
            st.subheader("Property Constraints")
            
            lipinski_compliance = st.checkbox(
                "Enforce Lipinski's Rule of Five",
                value=False,
                help="Ensure final molecules are drug-like"
            )
            
            max_rotatable_bonds = st.number_input(
                "Max Rotatable Bonds",
                min_value=0,
                max_value=20,
                value=10
            )
            
            rgroup_polarity = st.selectbox(
                "R-group Polarity Preference",
                ["Any", "Hydrophobic", "Hydrophilic", "Balanced"],
                index=0
            )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Sampling Temperature",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1
            )
            
            diversity_filter = st.checkbox(
                "Apply Diversity Filter",
                value=True,
                help="Remove highly similar R-group variants"
            )
            
            if diversity_filter:
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.7,
                    step=0.05
                )
        
        with col2:
            random_seed = st.number_input(
                "Random Seed (optional)",
                min_value=0,
                value=42
            )
            
            output_file = st.text_input(
                "Output File Name",
                value="rgroup_replacement_results.csv"
            )
            
            include_original = st.checkbox(
                "Include Original Molecules",
                value=True,
                help="Include input molecules in results for comparison"
            )
    
    # Generate button
    if st.button("🚀 Generate R-Group Variants", type="primary"):
        if not molecules:
            st.error("Please provide at least one molecule with R-group positions marked.")
        else:
            run_rgroup_replacement(
                molecules, model_file, device, num_variants,
                allowed_elements, max_rgroup_size, allow_rings_in_rgroup,
                lipinski_compliance, temperature, output_file
            )
    
    # Display results
    if 'rgroup_results' in st.session_state:
        show_rgroup_results(st.session_state.rgroup_results)

def run_rgroup_replacement(molecules, model_file, device, num_variants,
                          allowed_elements, max_rgroup_size, allow_rings,
                          lipinski_compliance, temperature, output_file):
    """Run R-group replacement"""
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Analyzing input molecules...")
        progress_bar.progress(0.2)
        
        # Validate molecules have R-group markers
        valid_molecules = []
        for mol in molecules:
            if '[*]' in mol or 'R' in mol:
                valid_molecules.append(mol)
        
        if not valid_molecules:
            st.error("No molecules with R-group markers ([*] or R) found.")
            return
        
        status_text.text("Generating R-group variants...")
        progress_bar.progress(0.7)
        
        # Simulate processing
        import time
        time.sleep(3)
        
        # Generate simulated results
        results_df = simulate_rgroup_results(valid_molecules, num_variants)
        
        progress_bar.progress(1.0)
        status_text.text("R-group replacement complete!")
        
        st.session_state.rgroup_results = {
            'dataframe': results_df,
            'output_file': output_file,
            'original_molecules': valid_molecules
        }
        
        st.success(f"✅ Successfully generated R-group variants for {len(valid_molecules)} molecules!")
        
    except Exception as e:
        st.error(f"❌ Error during R-group replacement: {str(e)}")

def simulate_rgroup_results(molecules, num_variants):
    """Simulate R-group replacement results"""
    
    np.random.seed(42)
    data = []
    
    # Sample R-groups for replacement
    r_groups = [
        "H", "C", "CC", "CCC", "CCCC", "C(C)C", "C(C)(C)C",
        "O", "OC", "OCC", "N", "NC", "NCC", "N(C)C",
        "F", "Cl", "Br", "CF3", "OCF3", "CN", "C=O", "C(=O)C",
        "c1ccccc1", "c1ccncc1", "c1cccnc1"
    ]
    
    for mol in molecules:
        for i in range(min(num_variants, 25)):  # Limit for demo
            # Replace R-group marker with random R-group
            r_group = np.random.choice(r_groups)
            if '[*]' in mol:
                variant = mol.replace('[*]', r_group)
            else:
                variant = mol.replace('R', r_group)
            
            nll = np.random.uniform(-5, -1)
            mw = np.random.uniform(150, 500)
            logp = np.random.uniform(-2, 5)
            tpsa = np.random.uniform(20, 120)
            rotatable_bonds = np.random.randint(0, 15)
            similarity = np.random.uniform(0.5, 0.95)
            
            data.append({
                'Original_Molecule': mol,
                'R_Group': r_group,
                'Generated_Molecule': variant,
                'NLL': nll,
                'Molecular_Weight': mw,
                'LogP': logp,
                'TPSA': tpsa,
                'Rotatable_Bonds': rotatable_bonds,
                'Similarity_to_Original': similarity,
                'Lipinski_Compliant': (mw <= 500 and logp <= 5 and tpsa <= 140),
                'Valid': np.random.choice([True, False], p=[0.85, 0.15])
            })
    
    return pd.DataFrame(data)

def show_rgroup_results(results):
    """Display R-group replacement results"""
    
    st.markdown('<div class="sub-header">📊 R-Group Replacement Results</div>', unsafe_allow_html=True)
    
    df = results['dataframe']
    original_molecules = results['original_molecules']
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Original Molecules", len(original_molecules))
    
    with col2:
        st.metric("Total Variants", len(df))
    
    with col3:
        valid_count = df['Valid'].sum() if 'Valid' in df.columns else len(df)
        st.metric("Valid Variants", valid_count)
    
    with col4:
        lipinski_count = df['Lipinski_Compliant'].sum() if 'Lipinski_Compliant' in df.columns else 0
        st.metric("Lipinski Compliant", lipinski_count)
    
    # Filter by original molecule
    if len(original_molecules) > 1:
        selected_molecule = st.selectbox(
            "Select Original Molecule:",
            original_molecules
        )
        
        filtered_df = df[df['Original_Molecule'] == selected_molecule]
        st.subheader(f"Variants for: {selected_molecule}")
    else:
        filtered_df = df
        st.subheader("All R-Group Variants")
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Visualizations
    if len(df) > 0:
        st.subheader("Property Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Property distribution
            if 'Molecular_Weight' in df.columns:
                fig = px.histogram(
                    df,
                    x='Molecular_Weight',
                    color='Lipinski_Compliant' if 'Lipinski_Compliant' in df.columns else None,
                    title="Molecular Weight Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Similarity vs LogP
            if 'Similarity_to_Original' in df.columns and 'LogP' in df.columns:
                fig = px.scatter(
                    df,
                    x='Similarity_to_Original',
                    y='LogP',
                    color='Valid' if 'Valid' in df.columns else None,
                    title="Similarity vs LogP",
                    hover_data=['R_Group']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Download results
    st.subheader("Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📄 Download CSV",
            csv_data,
            file_name=results['output_file'],
            mime="text/csv"
        )
    
    with col2:
        # Create summary report
        summary_report = create_rgroup_summary(df, original_molecules)
        st.download_button(
            "📊 Download Summary Report",
            summary_report,
            file_name="rgroup_summary.txt",
            mime="text/plain"
        )

def create_rgroup_summary(df, original_molecules):
    """Create a summary report for R-group replacement results"""
    
    report = f"""R-Group Replacement Summary Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

Input Summary:
- Original molecules: {len(original_molecules)}
- Total variants generated: {len(df)}

Quality Metrics:
- Valid molecules: {df['Valid'].sum() if 'Valid' in df.columns else 'N/A'}
- Lipinski compliant: {df['Lipinski_Compliant'].sum() if 'Lipinski_Compliant' in df.columns else 'N/A'}

Property Statistics:
"""
    
    if 'Molecular_Weight' in df.columns:
        report += f"- Molecular Weight: {df['Molecular_Weight'].mean():.1f} ± {df['Molecular_Weight'].std():.1f} Da\n"
    
    if 'LogP' in df.columns:
        report += f"- LogP: {df['LogP'].mean():.2f} ± {df['LogP'].std():.2f}\n"
    
    if 'TPSA' in df.columns:
        report += f"- TPSA: {df['TPSA'].mean():.1f} ± {df['TPSA'].std():.1f} Ų\n"
    
    if 'Similarity_to_Original' in df.columns:
        report += f"- Similarity to original: {df['Similarity_to_Original'].mean():.3f} ± {df['Similarity_to_Original'].std():.3f}\n"
    
    # Most common R-groups
    if 'R_Group' in df.columns:
        top_rgroups = df['R_Group'].value_counts().head(10)
        report += f"\nMost Common R-Groups:\n"
        for rgroup, count in top_rgroups.items():
            report += f"- {rgroup}: {count} occurrences\n"
    
    return report

def show_optimization_page():
    """Molecule optimization page"""
    
    st.markdown('<div class="sub-header">📈 Molecule Optimization</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Optimize existing molecules using reinforcement learning and multi-objective scoring functions.
    </div>
    """, unsafe_allow_html=True)
    
    # Optimization strategy selection
    st.subheader("Optimization Strategy")
    
    strategy = st.radio(
        "Select Optimization Strategy:",
        [
            "Reinforcement Learning (RL)",
            "Transfer Learning + RL",
            "Curriculum Learning (Staged RL)",
            "Multi-objective Optimization"
        ],
        help="Choose the optimization approach based on your requirements"
    )
    
    # Input molecules
    with st.expander("📥 Input Molecules", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            input_method = st.radio(
                "Input Method:",
                ["Upload File", "Text Input", "Database Query"]
            )
            
            molecules = []
            
            if input_method == "Upload File":
                uploaded_file = st.file_uploader(
                    "Upload Molecule File",
                    type=['smi', 'sdf', 'csv'],
                    help="File containing starting molecules"
                )
                
                if uploaded_file:
                    content = uploaded_file.read().decode('utf-8')
                    molecules = [line.strip() for line in content.split('\n') if line.strip()]
            
            elif input_method == "Text Input":
                molecules_text = st.text_area(
                    "Enter Starting Molecules (one per line)",
                    placeholder="CCO\nc1ccccc1\nCC(=O)O\n...",
                    height=150
                )
                
                if molecules_text:
                    molecules = [line.strip() for line in molecules_text.split('\n') if line.strip()]
            
            else:  # Database Query
                st.info("Database query functionality would connect to ChEMBL, PubChem, etc.")
                
                # Placeholder for database search
                query_text = st.text_input(
                    "Search Query:",
                    placeholder="aspirin, kinase inhibitor, etc."
                )
                
                if query_text:
                    # Simulate database results
                    molecules = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]  # Example results
        
        with col2:
            st.subheader("Generation Model")
            
            model_type = st.selectbox(
                "Generator Type:",
                ["Reinvent", "Mol2Mol", "LibInvent", "LinkInvent"]
            )
            
            model_file = st.text_input(
                f"{model_type} Model File",
                value=f"priors/{model_type.lower()}.prior"
            )
            
            if model_type == "Mol2Mol":
                similarity_strategy = st.selectbox(
                    "Optimization Strategy:",
                    ["Scaffold-based", "Pharmacophore-based", "Property-based"]
                )
    
    # Optimization targets
    with st.expander("🎯 Optimization Targets", expanded=True):
        st.subheader("Select Optimization Objectives")
        
        # Predefined target categories
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**ADMET Properties**")
            optimize_solubility = st.checkbox("Aqueous Solubility", value=False)
            optimize_permeability = st.checkbox("Membrane Permeability", value=False)
            optimize_clearance = st.checkbox("Hepatic Clearance", value=False)
            optimize_bioavailability = st.checkbox("Oral Bioavailability", value=False)
        
        with col2:
            st.markdown("**Physicochemical Properties**")
            optimize_mw = st.checkbox("Molecular Weight", value=True)
            optimize_logp = st.checkbox("LogP", value=True)
            optimize_tpsa = st.checkbox("TPSA", value=False)
            optimize_hbd = st.checkbox("H-Bond Donors", value=False)
            optimize_hba = st.checkbox("H-Bond Acceptors", value=False)
        
        with col3:
            st.markdown("**Activity & Selectivity**")
            optimize_activity = st.checkbox("Target Activity", value=False)
            optimize_selectivity = st.checkbox("Selectivity", value=False)
            optimize_toxicity = st.checkbox("Minimize Toxicity", value=False)
            optimize_synthetic = st.checkbox("Synthetic Accessibility", value=False)
        
        # Custom scoring components
        st.subheader("Custom Scoring Components")
        
        with st.expander("Add Custom Component"):
            component_type = st.selectbox(
                "Component Type:",
                ["QSAR Model", "Docking Score", "Custom Function", "Similarity"]
            )
            
            if component_type == "QSAR Model":
                qsar_target = st.text_input("Target Name:")
                qsar_model_file = st.file_uploader("Upload QSAR Model", type=['pkl', 'joblib'])
                qsar_desired_value = st.number_input("Desired Value:", value=0.0)
                qsar_weight = st.slider("Weight:", 0.0, 1.0, 0.5)
            
            elif component_type == "Docking Score":
                protein_file = st.file_uploader("Upload Protein Structure", type=['pdb'])
                binding_site = st.text_input("Binding Site (optional):")
                docking_software = st.selectbox("Docking Software:", ["AutoDock Vina", "Glide", "Gold"])
                desired_score = st.number_input("Desired Score:", value=-8.0)
            
            elif component_type == "Similarity":
                reference_smiles = st.text_input("Reference SMILES:")
                similarity_metric = st.selectbox("Metric:", ["Tanimoto", "Dice", "Cosine"])
                desired_similarity = st.slider("Desired Similarity:", 0.0, 1.0, 0.7)
    
    # Optimization parameters
    with st.expander("⚙️ Optimization Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Learning Parameters")
            
            num_steps = st.number_input(
                "Number of Optimization Steps",
                min_value=10,
                max_value=10000,
                value=1000,
                step=10
            )
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=10,
                max_value=1000,
                value=128
            )
            
            learning_rate = st.number_input(
                "Learning Rate",
                min_value=0.0001,
                max_value=0.1,
                value=0.0001,
                format="%.4f"
            )
        
        with col2:
            st.subheader("Generation Parameters")
            
            num_molecules_per_step = st.number_input(
                "Molecules per Step",
                min_value=10,
                max_value=1000,
                value=100
            )
            
            diversity_filter_threshold = st.slider(
                "Diversity Filter Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                help="Remove molecules too similar to previous generations"
            )
            
            inception_threshold = st.slider(
                "Inception Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                help="Threshold for adding molecules to memory"
            )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            device = st.selectbox("Compute Device", ["cuda:0", "cpu"])
            
            checkpoint_frequency = st.number_input(
                "Checkpoint Every N Steps",
                min_value=10,
                max_value=1000,
                value=100
            )
            
            tensorboard_logging = st.checkbox(
                "Enable TensorBoard Logging",
                value=True
            )
            
            if tensorboard_logging:
                tb_logdir = st.text_input(
                    "TensorBoard Log Directory",
                    value="tb_optimization"
                )
        
        with col2:
            random_seed = st.number_input("Random Seed (optional)", value=42)
            
            output_prefix = st.text_input(
                "Output File Prefix",
                value="optimization"
            )
            
            save_intermediate = st.checkbox(
                "Save Intermediate Results",
                value=True,
                help="Save results after each checkpoint"
            )
    
    # Run optimization
    if st.button("🚀 Start Optimization", type="primary"):
        if not molecules:
            st.error("Please provide starting molecules for optimization.")
        else:
            run_optimization(
                strategy, molecules, model_type, model_file,
                num_steps, batch_size, learning_rate,
                num_molecules_per_step, diversity_filter_threshold,
                device, output_prefix
            )
    
    # Display results
    if 'optimization_results' in st.session_state:
        show_optimization_results(st.session_state.optimization_results)

def run_optimization(strategy, molecules, model_type, model_file,
                    num_steps, batch_size, learning_rate,
                    num_molecules_per_step, diversity_threshold,
                    device, output_prefix):
    """Run molecule optimization"""
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Initializing optimization...")
        progress_bar.progress(0.1)
        
        # Create optimization configuration
        config = {
            "run_type": "staged_learning" if strategy == "Curriculum Learning (Staged RL)" else "transfer_learning",
            "device": device,
            "parameters": {
                "prior_file": model_file,
                "agent_file": model_file,
                "num_steps": num_steps,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "diversity_filter_threshold": diversity_threshold,
                "summary_csv_prefix": output_prefix
            }
        }
        
        # Add molecules file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False) as f:
            for mol in molecules:
                f.write(f"{mol}\n")
            molecules_file = f.name
        
        config["parameters"]["smiles_file"] = molecules_file
        
        status_text.text("Running optimization...")
        
        # Simulate optimization steps
        results_data = []
        for step in range(min(num_steps, 50)):  # Limit for demo
            progress = 0.1 + (step / 50) * 0.8
            progress_bar.progress(progress)
            status_text.text(f"Optimization step {step + 1}/{min(num_steps, 50)}")
            
            # Simulate step results
            step_results = simulate_optimization_step(molecules, step)
            results_data.extend(step_results)
            
            import time
            time.sleep(0.1)  # Simulate processing time
        
        progress_bar.progress(1.0)
        status_text.text("Optimization complete!")
        
        # Create results dataframe
        results_df = pd.DataFrame(results_data)
        
        st.session_state.optimization_results = {
            'dataframe': results_df,
            'config': config,
            'strategy': strategy,
            'starting_molecules': molecules,
            'output_prefix': output_prefix
        }
        
        st.success(f"✅ Optimization completed! Generated {len(results_df)} optimized molecules.")
        
        # Clean up
        os.unlink(molecules_file)
        
    except Exception as e:
        st.error(f"❌ Error during optimization: {str(e)}")

def simulate_optimization_step(starting_molecules, step):
    """Simulate one optimization step"""
    
    np.random.seed(42 + step)
    
    step_data = []
    num_generated = np.random.randint(5, 15)  # Vary number per step
    
    for i in range(num_generated):
        # Start with a random starting molecule and modify it
        base_mol = np.random.choice(starting_molecules)
        
        # Simulate score improvement over steps
        base_score = 0.3 + (step / 50) * 0.4 + np.random.normal(0, 0.1)
        base_score = max(0, min(1, base_score))  # Clamp to [0, 1]
        
        # Generate molecular properties
        mw = np.random.uniform(200, 500)
        logp = np.random.uniform(1, 4)
        tpsa = np.random.uniform(40, 120)
        
        # Simulate improved molecule (just modify the SMILES slightly for demo)
        generated_mol = base_mol + "C" if len(base_mol) < 20 else base_mol
        
        step_data.append({
            'Step': step,
            'Starting_Molecule': base_mol,
            'Generated_Molecule': generated_mol,
            'Total_Score': base_score,
            'MW_Score': max(0, 1 - abs(mw - 350) / 150),  # Target MW ~350
            'LogP_Score': max(0, 1 - abs(logp - 2.5) / 2),  # Target LogP ~2.5
            'TPSA_Score': max(0, 1 - abs(tpsa - 80) / 40),  # Target TPSA ~80
            'Molecular_Weight': mw,
            'LogP': logp,
            'TPSA': tpsa,
            'Valid': np.random.choice([True, False], p=[0.9, 0.1]),
            'NLL': np.random.uniform(-4, -1)
        })
    
    return step_data

def show_optimization_results(results):
    """Display optimization results with progress tracking"""
    
    st.markdown('<div class="sub-header">📊 Optimization Results</div>', unsafe_allow_html=True)
    
    df = results['dataframe']
    strategy = results['strategy']
    starting_molecules = results['starting_molecules']
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Starting Molecules", len(starting_molecules))
    
    with col2:
        st.metric("Generated Molecules", len(df))
    
    with col3:
        max_score = df['Total_Score'].max() if 'Total_Score' in df.columns else 0
        st.metric("Best Score", f"{max_score:.3f}")
    
    with col4:
        final_step = df['Step'].max() if 'Step' in df.columns else 0
        st.metric("Final Step", final_step)
    
    # Progress visualization
    if 'Step' in df.columns and 'Total_Score' in df.columns:
        st.subheader("Optimization Progress")
        
        # Score progression
        step_scores = df.groupby('Step')['Total_Score'].agg(['mean', 'max', 'std']).reset_index()
        
        fig = go.Figure()
        
        # Average score
        fig.add_trace(go.Scatter(
            x=step_scores['Step'],
            y=step_scores['mean'],
            mode='lines+markers',
            name='Average Score',
            line=dict(color='blue')
        ))
        
        # Best score
        fig.add_trace(go.Scatter(
            x=step_scores['Step'],
            y=step_scores['max'],
            mode='lines+markers',
            name='Best Score',
            line=dict(color='red')
        ))
        
        # Error bars
        fig.add_trace(go.Scatter(
            x=step_scores['Step'],
            y=step_scores['mean'] + step_scores['std'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=step_scores['Step'],
            y=step_scores['mean'] - step_scores['std'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            name='±1 Std Dev',
            fillcolor='rgba(0,100,80,0.2)'
        ))
        
        fig.update_layout(
            title="Score Evolution During Optimization",
            xaxis_title="Optimization Step",
            yaxis_title="Score",
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Best molecules
    st.subheader("Top Optimized Molecules")
    
    if 'Total_Score' in df.columns:
        top_molecules = df.nlargest(10, 'Total_Score')
        st.dataframe(top_molecules, use_container_width=True)
    
    # Property analysis
    st.subheader("Property Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Molecular_Weight' in df.columns and 'LogP' in df.columns:
            fig = px.scatter(
                df,
                x='Molecular_Weight',
                y='LogP',
                color='Total_Score' if 'Total_Score' in df.columns else None,
                title="Property Space Exploration",
                hover_data=['Generated_Molecule'] if 'Generated_Molecule' in df.columns else None
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'Total_Score' in df.columns:
            fig = px.histogram(
                df,
                x='Total_Score',
                title="Score Distribution",
                nbins=20
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Download options
    st.subheader("Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📄 Download Full Results",
            csv_data,
            file_name=f"{results['output_prefix']}_full_results.csv",
            mime="text/csv"
        )
    
    with col2:
        if 'Total_Score' in df.columns:
            top_molecules = df.nlargest(100, 'Total_Score')
            top_csv = top_molecules.to_csv(index=False)
            st.download_button(
                "🏆 Download Top 100",
                top_csv,
                file_name=f"{results['output_prefix']}_top100.csv",
                mime="text/csv"
            )
    
    with col3:
        config_data = json.dumps(results['config'], indent=2)
        st.download_button(
            "⚙️ Download Config",
            config_data,
            file_name=f"{results['output_prefix']}_config.json",
            mime="application/json"
        )

# Additional placeholder functions to complete the application structure
def show_library_page():
    """Library design page placeholder"""
    st.markdown('<div class="sub-header">📚 Library Design</div>', unsafe_allow_html=True)
    st.info("Library design functionality - to be implemented with enumeration and selection strategies")

def show_scoring_page():
    """Scoring functions configuration page placeholder"""
    st.markdown('<div class="sub-header">🎯 Scoring Functions</div>', unsafe_allow_html=True)
    st.info("Scoring function configuration - to be implemented with component builder")

def show_transfer_learning_page():
    """Transfer learning page placeholder"""
    st.markdown('<div class="sub-header">🎓 Transfer Learning</div>', unsafe_allow_html=True)
    st.info("Transfer learning configuration - to be implemented")

def show_reinforcement_learning_page():
    """Reinforcement learning page placeholder"""
    st.markdown('<div class="sub-header">💪 Reinforcement Learning</div>', unsafe_allow_html=True)
    st.info("Reinforcement learning configuration - to be implemented")

def show_visualization_page():
    """Results visualization page placeholder"""
    st.markdown('<div class="sub-header">📊 Results Visualization</div>', unsafe_allow_html=True)
    st.info("Advanced visualization and analysis tools - to be implemented")

def show_config_page():
    """Configuration manager page placeholder"""
    st.markdown('<div class="sub-header">⚙️ Configuration Manager</div>', unsafe_allow_html=True)
    st.info("Configuration templates and management - to be implemented")

def show_documentation_page():
    """Documentation page placeholder"""
    st.markdown('<div class="sub-header">📖 Documentation</div>', unsafe_allow_html=True)
    st.info("Help and documentation - to be implemented")
