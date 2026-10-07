AddReference "C:\Program Files\Autodesk\Inventor Nastran 2025\In-CAD\System\NINIlogic.dll"

Sub Main()

    '=========================================================
    ' USER SETTINGS
    '=========================================================

    ' Inventor Nastran 2025 installation folder
    Dim NastranSystemFolder As String
    NastranSystemFolder = _
        "C:\Program Files\Autodesk\Inventor Nastran 2025\In-CAD\System"

    ' Export folder:
    ' <Inventor file folder>\NastranExports\
    '
    ' Example:
    ' C:\Project\Part1.ipt
    ' C:\Project\NastranExports\Part1.nas

    '=========================================================
    ' GET ACTIVE INVENTOR DOCUMENT
    '=========================================================

    Dim oDoc As Document
    oDoc = ThisDoc.Document

    If oDoc Is Nothing Then
        Throw New Exception("No active Inventor document.")
    End If

    ' This rule is intended for parts
    If Not TypeOf oDoc Is PartDocument Then
        Throw New Exception( _
            "This rule is intended for an Inventor part (.ipt).")
    End If

    ' The document must have been saved
    If String.IsNullOrEmpty(oDoc.FullFileName) Then
        Throw New Exception( _
            "Save the Inventor part before running the Nastran automation.")
    End If

    '=========================================================
    ' FILE/FOLDER INFORMATION
    '=========================================================

    Dim ModelFolder As String
    Dim ModelName As String
    Dim FeaFolder As String
    Dim ExportFolder As String
    Dim ExportFile As String

    ModelFolder = System.IO.Path.GetDirectoryName(oDoc.FullFileName)

    ModelName = _
        System.IO.Path.GetFileNameWithoutExtension(oDoc.FullFileName)

    ' Inventor Nastran normally stores generated files here:
    '
    ' <model folder>\<model name>\InCAD\FEA\
    '
    FeaFolder = System.IO.Path.Combine( _
        ModelFolder, _
        ModelName, _
        "InCAD", _
        "FEA")

    ExportFolder = System.IO.Path.Combine( _
        ModelFolder, _
        "NastranExports")

    ExportFile = System.IO.Path.Combine( _
        ExportFolder, _
        ModelName & ".nas")

    ' Create export folder if necessary
    System.IO.Directory.CreateDirectory(ExportFolder)

    '=========================================================
    ' CHECK NASTRAN INSTALLATION
    '=========================================================

    Dim ExternalRuleFolder As String
    ExternalRuleFolder = _
        System.IO.Path.Combine(NastranSystemFolder, "iLogic")

    If Not System.IO.Directory.Exists(NastranSystemFolder) Then
        Throw New Exception( _
            "Inventor Nastran 2025 was not found at:" & vbCrLf & vbCrLf & _
            NastranSystemFolder & vbCrLf & vbCrLf & _
            "Change NastranSystemFolder at the top of the rule.")
    End If

    If Not System.IO.Directory.Exists(ExternalRuleFolder) Then
        Throw New Exception( _
            "The Nastran iLogic folder was not found:" & vbCrLf & vbCrLf & _
            ExternalRuleFolder)
    End If

    '=========================================================
    ' CONFIGURE INVENTOR ILOGIC FOR NASTRAN
    '=========================================================

    Dim oIlogic As IiLogicAutomation
    oIlogic = iLogicVb.Automation

    Dim ExternalRuleDirectory(0) As String
    ExternalRuleDirectory(0) = ExternalRuleFolder

    oIlogic.FileOptions.ExternalRuleDirectories = _
        ExternalRuleDirectory

    oIlogic.FileOptions.AddinDirectory = _
        NastranSystemFolder

    '=========================================================
    ' REMEMBER WHEN WE STARTED
    '=========================================================

    Dim StartTimeUtc As DateTime
    StartTimeUtc = DateTime.UtcNow

    '=========================================================
    ' FORCE INVENTOR MODEL UPDATE
    '=========================================================

    oDoc.Update()

    '=========================================================
    ' ENTER INVENTOR NASTRAN ENVIRONMENT
    '=========================================================

    iLogicVb.RunExternalRule("IlogicExtRule_EnterInCADEnv")

    '=========================================================
    ' CONNECT TO INVENTOR NASTRAN
    '=========================================================

    Dim AddinObj As New NINIlogic.InCadAddin

    '=========================================================
    ' GENERATE / REGENERATE MESH
    '=========================================================

    Dim strCmdForMesh As String
    strCmdForMesh = "<GenerateMesh/>"

    Dim MeshResult As Integer
    MeshResult = AddinObj.Run(strCmdForMesh)

    If MeshResult <> 0 Then
        Throw New Exception( _
            "Inventor Nastran failed to generate the mesh. " & _
            "Command result: " & MeshResult)
    End If

    iLogicVb.UpdateWhenDone = True

    '=========================================================
    ' WAIT FOR THE .NAS FILE TO BE UPDATED
    '=========================================================

    Dim NewestNasFile As String
    NewestNasFile = ""

    Dim WaitTime As Integer
    WaitTime = 0

    Do While WaitTime < 60000

        If System.IO.Directory.Exists(FeaFolder) Then

            NewestNasFile = FindNewestNasFile( _
                FeaFolder, _
                StartTimeUtc)

            If NewestNasFile <> "" Then
                Exit Do
            End If

        End If

        System.Threading.Thread.Sleep(500)
        WaitTime = WaitTime + 500

    Loop

    '=========================================================
    ' CHECK THAT A NAS FILE WAS CREATED/UPDATED
    '=========================================================

    If NewestNasFile = "" Then

        Throw New Exception( _
            "The mesh was generated, but no newly updated .nas file was found." & _
            vbCrLf & vbCrLf & _
            "Check that 'Auto Update Nastran File' is enabled in:" & _
            vbCrLf & _
            "Nastran > Default Settings > General > Nastran File Settings" & _
            vbCrLf & vbCrLf & _
            "Expected folder:" & vbCrLf & _
            FeaFolder)

    End If

    '=========================================================
    ' COPY NAS FILE TO EXPORT LOCATION
    '=========================================================

    Try

        System.IO.File.Copy( _
            NewestNasFile, _
            ExportFile, _
            True)

    Catch ex As Exception

        Throw New Exception( _
            "The Nastran file was generated, but copying it failed." & _
            vbCrLf & vbCrLf & _
            "Source:" & vbCrLf & _
            NewestNasFile & _
            vbCrLf & vbCrLf & _
            "Destination:" & vbCrLf & _
            ExportFile & _
            vbCrLf & vbCrLf & _
            "Error:" & vbCrLf & _
            ex.Message, ex)

    End Try

End Sub


'=====================================================================
' FIND THE NEWEST .NAS FILE THAT WAS CREATED/UPDATED AFTER StartTimeUtc
'=====================================================================

Function FindNewestNasFile( _
    Folder As String, _
    StartTimeUtc As DateTime) As String

    Dim Files() As String

    Try

        Files = System.IO.Directory.GetFiles( _
            Folder, _
            "*.nas")

    Catch

        Return ""

    End Try

    Dim NewestFile As String
    NewestFile = ""

    Dim NewestTime As DateTime
    NewestTime = StartTimeUtc

    For Each FileName As String In Files

        Try

            Dim FileTime As DateTime
            FileTime = _
                System.IO.File.GetLastWriteTimeUtc(FileName)

            If FileTime >= NewestTime Then

                NewestTime = FileTime
                NewestFile = FileName

            End If

        Catch

            ' Ignore files that cannot be inspected

        End Try

    Next

    Return NewestFile

End Function