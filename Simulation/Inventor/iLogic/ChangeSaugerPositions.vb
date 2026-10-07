' External iLogic rule: update the four named Inventor parameters.
' Expected arguments: s1_x, s1_z, s2_x, s2_z.

Dim parameterNames() As String = {"s1_x", "s1_z", "s2_x", "s2_z"}

For Each parameterName As String In parameterNames
    If Not RuleArguments.Exists(parameterName) Then
        Throw New Exception("Missing iLogic argument: " & parameterName)
    End If

    Parameter(parameterName) = CDbl(RuleArguments.Value(parameterName))
Next

InventorVb.DocumentUpdate()
