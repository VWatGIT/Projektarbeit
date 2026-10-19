' External iLogic rule: update all named Inventor sauger parameters.
' Expected arguments: sauger_count, s1_x, s1_z, s2_x, s2_z, ...

If Not RuleArguments.Exists("sauger_count") Then
    Throw New Exception("Missing iLogic argument: sauger_count")
End If

Dim saugerCount As Integer = CInt(RuleArguments.Value("sauger_count"))
If saugerCount < 1 Then
    Throw New Exception("sauger_count must be positive.")
End If

For saugerIndex As Integer = 1 To saugerCount
    For Each axis As String In New String() {"x", "z"}
        Dim parameterName As String = "s" & saugerIndex & "_" & axis
        If Not RuleArguments.Exists(parameterName) Then
            Throw New Exception("Missing iLogic argument: " & parameterName)
        End If

        Parameter(parameterName) = CDbl(RuleArguments.Value(parameterName))
    Next
Next

InventorVb.DocumentUpdate()
