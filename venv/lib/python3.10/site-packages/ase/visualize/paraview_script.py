# fmt: off

def main():
    import paraview.simple as para
    version_major = para.servermanager.vtkSMProxyManager.GetVersionMajor()
    source = para.GetActiveSource()
    renderView1 = para.GetRenderView()
    atoms = para.Glyph(
        Input=source,
        GlyphType='Sphere',
        Scalars='radii',
        ScaleMode='scalar',
    )
    para.RenameSource('Atoms', atoms)
    atomsDisplay = para.Show(atoms, renderView1)
    if version_major <= 4:
        atoms.SetScaleFactor = 0.8
        atomicnumbers_PVLookupTable = para.GetLookupTableForArray(
            'atomic numbers', 1)
        atomsDisplay.ColorArrayName = ('POINT_DATA', 'atomic numbers')
        atomsDisplay.LookupTable = atomicnumbers_PVLookupTable
    else:
        atoms.ScaleFactor = 0.8
        para.ColorBy(atomsDisplay, 'atomic numbers')
        atomsDisplay.SetScalarBarVisibility(renderView1, True)
    para.Render()


if __name__ == '__main__':
    main()
